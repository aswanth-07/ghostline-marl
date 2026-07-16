from __future__ import annotations

import asyncio
import importlib
import itertools
import math
import random
import time
from typing import Any, Literal

import numpy as np
import pygame

from ghostline.audio import AudioDirector
from ghostline.config import TIERS
from ghostline.policies import FairScriptedPolicy
from ghostline.presentation import (
    GhostlineRenderer,
    TOUCH_DASH_CENTER,
    TOUCH_DASH_RADIUS,
    TOUCH_JOYSTICK_CENTER,
    TOUCH_PAUSE_RECT,
    TOUCH_PULSE_CENTER,
    TOUCH_PULSE_RADIUS,
)
from ghostline.resources import runtime_asset_path
from ghostline.progression import (
    DEFAULT_BINDINGS,
    load_progression,
    load_settings,
    record_run,
    record_success,
    save_settings,
)
from ghostline.simulation import GhostlineSimulation
from ghostline.types import Action

BRIEFINGS = {
    1: ("Learn the line.", "Acquire the contract quota and leave through the green extraction relay."),
    2: ("The building is watching.", "Camera cones raise TRACE. Break sight and let the signal cool."),
    3: ("Patrols are live.", "Guards hear noisy dashes, investigate disturbances, and share alerts."),
    4: ("Carry a countermeasure.", "Your pulse disables electronics and jams guard radios."),
    5: ("Lockdown teams are standing by.", "High TRACE deploys a response drone. Optional data is valuable, not safe."),
    6: ("No route repeats.", "Every system is live. Take the quota, keep your integrity, disappear."),
}

MOVE_LABELS = ("HOLD", "N", "NE", "E", "SE", "S", "SW", "W", "NW")
CONTROL_ACTIONS = (
    ("move_up", "MOVE UP"),
    ("move_down", "MOVE DOWN"),
    ("move_left", "MOVE LEFT"),
    ("move_right", "MOVE RIGHT"),
    ("dash", "DASH"),
    ("pulse", "PULSE"),
    ("restart", "RESTART"),
    ("pause", "PAUSE / MENU"),
    ("menu_up", "MENU UP"),
    ("menu_down", "MENU DOWN"),
    ("confirm", "MENU CONFIRM"),
    ("back", "MENU BACK"),
)

# Curated held-out validation contracts for the public "Watch Agent" path.
# They make the first showcase representative without consuming or publishing
# another final-test slice. Manual seed editing and random human play remain
# available for honest failure inspection.
AGENT_SHOWCASE_SEEDS = {
    1: 1_007_004,
    2: 1_015_005,
    3: 1_023_012,
    4: 1_031_013,
    5: 1_039_028,
    6: 1_047_023,
}
PORTFOLIO_DEMO_TIER = 6
PORTFOLIO_DEMO_SEED = 2_000_000


def mission_score(sim: GhostlineSimulation) -> int:
    if not sim.extracted:
        return 0
    score = sim.data * 900 + int(sim.remaining_seconds * 18)
    score += sim.integrity * 700
    score += max(0, 2500 - int(sim.max_trace * 20))
    score -= sim.damage_taken * 500 + sim.detections * 80
    return max(0, score)


def apply_human_timer_assist(sim: GhostlineSimulation, enabled: bool) -> None:
    """Extend only an explicitly assisted human contract; RL contracts stay unchanged."""
    if enabled:
        sim.level.mission_seconds = int(round(sim.level.mission_seconds * 1.35))


class GameApp:
    def __init__(
        self,
        *,
        initial_tier: int | None = None,
        seed: int | None = None,
        mode: Literal["menu", "play", "lab"] = "menu",
    ):
        self.progression = load_progression()
        self.settings = load_settings()
        self.selected_tier = initial_tier or int(self.progression["highest_unlocked_tier"])
        self.seed = seed
        self.lab_seed = seed if seed is not None else AGENT_SHOWCASE_SEEDS[self.selected_tier]
        self._lab_seed_is_custom = seed is not None
        self.mode = mode
        self.state = "main" if mode == "menu" else ("lab" if mode == "lab" else "briefing")
        self.selection = self.selected_tier - 1 if mode == "lab" else 0
        self.running = True
        self.fullscreen = bool(self.settings["display"]["fullscreen"])
        self.sim = GhostlineSimulation(seed=seed or 10101, tier=self.selected_tier)
        self.renderer = GhostlineRenderer(
            self.sim,
            visible=True,
            screen_shake=bool(self.settings["display"]["screen_shake"]),
            accessibility=self.settings["accessibility"],
        )
        if self.fullscreen:
            self.renderer.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.audio = AudioDirector(enabled=bool(self.settings["audio"]["enabled"]))
        self._apply_audio_settings()
        self.policy = FairScriptedPolicy()
        self.learned_policy = None
        self.policy_name = "FAIR SCRIPTED BASELINE"
        self.agent_env = None
        self.agent_hidden = None
        self._load_runtime_policy()
        self._agent_action = 0
        self._agent_tick = 0
        self._agent_latency_ms = 0.0
        self._rebind_action: str | None = None
        self._settings_return_state = "settings"
        self._debrief_agent = False
        self._telemetry: dict[str, Any] = {}
        self.touch_controls_enabled = False
        self._touch_points: dict[int, tuple[float, float]] = {}
        self._touch_roles: dict[int, str] = {}

    def _load_runtime_policy(self) -> None:
        # Resolve relative to the source tree, installed wheel, or PyInstaller
        # bundle. The old working-directory lookup silently dropped Agent Lab
        # to the weaker scripted fallback when launched from another folder.
        with runtime_asset_path("models/ghostline-policy.onnx") as onnx_path:
            try:
                if onnx_path is not None:
                    adapter = getattr(importlib.import_module("ghostline.inference"), "OnnxGhostlinePolicy")
                    self.learned_policy = adapter(onnx_path)
                    self.policy_name = "RECURRENT ONNX POLICY"
                    return
            except Exception:
                # A missing or incompatible policy must never prevent human play.
                self.learned_policy = None
        for relative in ("models/ghostline-policy.pt", "artifacts/torchrl/champion/best.pt"):
            with runtime_asset_path(relative) as candidate:
                if candidate is None:
                    continue
                try:
                    load_policy = getattr(importlib.import_module("ghostline.model"), "load_policy")
                    self.learned_policy = load_policy(candidate)
                    self.policy_name = "RECURRENT PYTORCH POLICY"
                    return
                except Exception:
                    self.learned_policy = None
        self.policy_name = "EXPERT FALLBACK // MODEL UNAVAILABLE"

    def run(self) -> int:
        try:
            while self.running:
                self.tick()
            return 0
        finally:
            if self.agent_env is not None:
                self.agent_env.close()
            self.audio.close()
            self.renderer.close()

    async def run_async(self) -> int:
        """Browser-compatible cooperative game loop used by pygbag."""

        try:
            while self.running:
                self.tick()
                await asyncio.sleep(0)
            return 0
        finally:
            if self.agent_env is not None:
                self.agent_env.close()
            self.audio.close()
            self.renderer.close()

    def tick(self) -> None:
        # The procedural score belongs to active contracts, not menus or pause
        # screens. Browser/tab focus is tracked independently by AudioDirector.
        self.audio.set_gameplay_active(self.state in {"play", "lab_play"})
        handlers = {
            "main": self._main_menu,
            "stage_select": self._stage_select,
            "briefing": self._briefing,
            "play": lambda: self._play(agent=False),
            "lab": self._lab_select,
            "lab_play": lambda: self._play(agent=True),
            "pause": self._pause,
            "howto": self._how_to,
            "settings": self._settings,
            "settings_audio": self._settings_audio,
            "settings_accessibility": self._settings_accessibility,
            "settings_display": self._settings_display,
            "settings_controls": self._settings_controls,
            "credits": self._credits,
            "debrief": self._debrief,
        }
        handlers[self.state]()

    def _key(self, action: str) -> int:
        name = str(self.settings["bindings"].get(action, DEFAULT_BINDINGS[action]))
        try:
            return pygame.key.key_code(name)
        except (ValueError, TypeError):
            return pygame.key.key_code(DEFAULT_BINDINGS[action])

    def _key_label(self, action: str) -> str:
        return str(self.settings["bindings"].get(action, DEFAULT_BINDINGS[action])).upper()

    def _events(self) -> list[pygame.event.Event]:
        events = list(pygame.event.get())
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == getattr(pygame, "WINDOWFOCUSLOST", -999):
                self.audio.set_focus_active(False)
                if self.state == "play":
                    self.state, self.selection = "pause", 0
            elif event.type == getattr(pygame, "WINDOWFOCUSGAINED", -998):
                self.audio.set_focus_active(True)
        return events

    def _event_window_position(self, event: pygame.event.Event) -> tuple[float, float] | None:
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            position = getattr(event, "pos", None)
            return tuple(position) if position is not None else None
        if event.type in (pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP):
            width, height = self.renderer.window.get_size()
            return float(event.x) * width, float(event.y) * height
        return None

    def _menu_events(
        self,
        events: list[pygame.event.Event],
        count: int,
        *,
        horizontal: bool = False,
        compact: bool = False,
        pointer_count: int | None = None,
        pointer_offset: int = 0,
    ) -> str | None:
        hit_count = count if pointer_count is None else pointer_count
        for event in events:
            if event.type in (pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP):
                self.touch_controls_enabled = True
            if event.type in (pygame.MOUSEMOTION, pygame.FINGERMOTION):
                position = self._event_window_position(event)
                local_hover = (
                    self.renderer.menu_item_at(position, hit_count, compact=compact)
                    if position is not None
                    else None
                )
                hovered = local_hover + pointer_offset if local_hover is not None else None
                if hovered is not None and hovered != self.selection:
                    self.selection = hovered
                    self.audio.menu_move()
                if event.type == pygame.MOUSEMOTION:
                    try:
                        pygame.mouse.set_cursor(
                            pygame.SYSTEM_CURSOR_HAND if hovered is not None else pygame.SYSTEM_CURSOR_ARROW
                        )
                    except pygame.error:
                        pass
                continue
            if event.type in (pygame.MOUSEBUTTONUP, pygame.FINGERUP):
                if event.type == pygame.MOUSEBUTTONUP and int(getattr(event, "button", 0)) != 1:
                    continue
                position = self._event_window_position(event)
                local_click = (
                    self.renderer.menu_item_at(position, hit_count, compact=compact)
                    if position is not None
                    else None
                )
                clicked = local_click + pointer_offset if local_click is not None else None
                if clicked is not None:
                    self.selection = clicked
                    self.audio.menu_confirm()
                    return "confirm"
                continue
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (self._key("menu_up"), pygame.K_UP):
                self.selection = (self.selection - 1) % count
                self.audio.menu_move()
            elif event.key in (self._key("menu_down"), pygame.K_DOWN):
                self.selection = (self.selection + 1) % count
                self.audio.menu_move()
            elif horizontal and event.key == pygame.K_LEFT:
                self.audio.menu_move()
                return "left"
            elif horizontal and event.key == pygame.K_RIGHT:
                self.audio.menu_move()
                return "right"
            elif event.key == self._key("confirm"):
                self.audio.menu_confirm()
                return "confirm"
            elif event.key == self._key("back"):
                return "back"
        return None

    def _main_menu(self) -> None:
        items = ["PLAY CONTRACTS", "AGENT LAB", "HOW TO PLAY", "SETTINGS", "CREDITS", "QUIT"]
        choice = self._menu_events(self._events(), len(items))
        if choice == "confirm":
            self.state = ("stage_select", "lab", "howto", "settings", "credits", "quit")[self.selection]
            self.selection = 0
            if self.state == "lab" and not self._lab_seed_is_custom:
                # The first Watch Agent run is the exact tier-six contract used
                # by the portfolio recording. It makes the live and recorded
                # policy directly comparable instead of silently showing two
                # different procedural seeds.
                self.selection = PORTFOLIO_DEMO_TIER - 1
                self.lab_seed = PORTFOLIO_DEMO_SEED
            if self.state == "quit":
                self.running = False
        cleared = len(self.progression.get("best_scores", {}))
        self.renderer.draw_screen(
            title="GHOSTLINE",
            subtitle="Move unseen. Take the signal. Leave no trace.",
            items=items,
            selected=self.selection,
            badge="PROCEDURAL STEALTH // RL SHOWCASE",
            panel=[
                "OPERATIVE STATUS",
                f"CLEARANCE       {self.progression['highest_unlocked_tier']}/6",
                f"CONTRACTS WON   {cleared}/6",
                f"RUNTIME POLICY  {self.policy_name}",
                "",
                "ONE WORLD. TWO CONTROLLERS.",
                "IDENTICAL SENSORS AND RULES.",
            ],
            footer=f"{self._key_label('menu_up')}/{self._key_label('menu_down')}  NAVIGATE     {self._key_label('confirm')}  SELECT",
        )

    def _stage_select(self) -> None:
        unlocked = int(self.progression["highest_unlocked_tier"])
        items = [f"TIER {tier}  {TIERS[tier].name.upper()}{'' if tier <= unlocked else '  [LOCKED]'}" for tier in range(1, 7)]
        choice = self._menu_events(self._events(), len(items))
        if choice == "back":
            self.state, self.selection = "main", 0
        elif choice == "confirm":
            tier = self.selection + 1
            if tier <= unlocked:
                self.selected_tier = tier
                self.state = "briefing"
        spec = TIERS[self.selection + 1]
        best = int(self.progression.get("best_scores", {}).get(str(spec.number), 0))
        self.renderer.draw_screen(
            title="CONTRACTS",
            subtitle=f"CLEARANCE {unlocked}/6",
            items=items,
            selected=self.selection,
            panel=[
                f"TIER {spec.number} // {spec.name.upper()}",
                f"FACILITY   {spec.room_columns}x{spec.room_rows} MODULES",
                f"QUOTA      {spec.quota} DATA",
                f"SECURITY   {spec.cameras} CAM / {spec.guards} GUARD",
                f"WINDOW     {spec.mission_seconds // 60}:{spec.mission_seconds % 60:02d}",
                f"BEST       {best:06d}" if best else "BEST       —",
            ],
            footer=f"{self._key_label('back')}  BACK     COMPLETE A CONTRACT TO UNLOCK THE NEXT TIER",
        )

    def _lab_select(self) -> None:
        items = [f"WATCH TIER {tier}  {TIERS[tier].name.upper()}" for tier in range(1, 7)]
        previous_tier = self.selection + 1
        choice = self._menu_events(self._events(), len(items), horizontal=True)
        selected_tier = self.selection + 1
        if selected_tier != previous_tier and not self._lab_seed_is_custom:
            self.lab_seed = AGENT_SHOWCASE_SEEDS[selected_tier]
        if choice == "back":
            self.state, self.selection = "main", 0
        elif choice == "confirm":
            self.selected_tier = self.selection + 1
            self._start_mission(agent=True, replay_seed=self.lab_seed)
        elif choice in ("left", "right"):
            modifiers = pygame.key.get_mods()
            step = 100 if modifiers & pygame.KMOD_SHIFT else 1
            self.lab_seed = max(0, self.lab_seed + (step if choice == "right" else -step))
            self._lab_seed_is_custom = True
        panel = self._lab_history_panel(self.selection + 1)
        if selected_tier == PORTFOLIO_DEMO_TIER and self.lab_seed == PORTFOLIO_DEMO_SEED and not self._lab_seed_is_custom:
            seed_label = "PORTFOLIO VIDEO REPLAY"
        else:
            seed_label = "CUSTOM REPLAY SEED" if self._lab_seed_is_custom else "SHOWCASE VALIDATION SEED"
        self.renderer.draw_screen(
            title="AGENT LAB",
            subtitle="Player-equivalent // deterministic replay",
            items=items,
            selected=self.selection,
            badge=self.policy_name,
            panel=[f"{seed_label}  {self.lab_seed}", "PUBLIC SENSORS // NO HIDDEN STATE", "LEFT/RIGHT ±1", "SHIFT + LEFT/RIGHT ±100", "", *panel],
            footer=f"{self._key_label('confirm')}  WATCH     LEFT/RIGHT  SEED     {self._key_label('back')}  BACK",
        )

    def _lab_history_panel(self, tier: int) -> list[str]:
        runs = [run for run in self.progression.get("recent_runs", []) if int(run.get("tier", -1)) == tier]
        human = next((run for run in reversed(runs) if run.get("controller") == "human"), None)
        agent = next((run for run in reversed(runs) if run.get("controller") == "agent"), None)

        def line(label: str, run: dict[str, Any] | None) -> str:
            if run is None:
                return f"{label:<7} NO LOCAL RUN"
            status = "CLEAR" if run.get("success") else "FAIL"
            return f"{label:<7} {status} {float(run.get('duration_seconds', 0)):5.1f}s T{float(run.get('max_trace', 0)):04.1f}"

        return ["MATCHED LOCAL RESULTS", line("HUMAN", human), line("AGENT", agent)]

    def _briefing(self) -> None:
        events = self._events()
        title, directive = BRIEFINGS[self.selected_tier]
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == self._key("confirm"):
                self.audio.menu_confirm()
                self._start_mission(agent=False)
            elif event.type == pygame.KEYDOWN and event.key == self._key("back"):
                self.state = "stage_select"
        spec = TIERS[self.selected_tier]
        self.renderer.draw_screen(
            title=f"TIER {spec.number}: {spec.name.upper()}",
            subtitle=title,
            body=[
                directive,
                "",
                f"FACILITY  {spec.room_columns}x{spec.room_rows} MODULES",
                f"CONTRACT QUOTA  {spec.quota} DATA",
                f"SECURITY  {spec.cameras} CAMERAS / {spec.guards} GUARDS",
                f"WINDOW  {spec.mission_seconds // 60}:{spec.mission_seconds % 60:02d}",
            ],
            badge=f"CONTRACT // {self.selected_tier:02d}",
            panel=[
                "FIELD PROTOCOL",
                "AMBER   DATA TERMINAL",
                "GREEN   EXTRACTION RELAY",
                "CONE    ACTIVE SIGHTLINE",
                "I/II/III  PATROL GRADE",
                "TRACE   NETWORK PRESSURE",
                "",
                f"SEED NAMESPACE  {'GUIDED' if self.selected_tier == 1 else 'PROCEDURAL'}",
            ],
            footer=f"{self._key_label('confirm')}  DEPLOY     {self._key_label('back')}  CONTRACTS",
        )

    def _start_mission(self, *, agent: bool, replay_seed: int | None = None) -> None:
        self._touch_points.clear()
        self._touch_roles.clear()
        if self.agent_env is not None:
            self.agent_env.close()
            self.agent_env = None
        if replay_seed is not None:
            mission_seed = replay_seed
        elif self.seed is not None:
            mission_seed = self.seed
        elif self.selected_tier == 1 and not self.progression["best_scores"]:
            mission_seed = 10101
        else:
            mission_seed = random.SystemRandom().randrange(0, 1_000_000)
        self.last_mission_seed = mission_seed
        if agent and self.learned_policy is not None:
            GhostlineEnv = getattr(importlib.import_module("ghostline.env"), "GhostlineEnv")
            self.agent_env = GhostlineEnv(seed=mission_seed, tier=self.selected_tier)
            self.agent_observation, _ = self.agent_env.reset(seed=mission_seed)
            self.sim = self.agent_env.sim
            self.agent_hidden = None
        else:
            self.sim.reset(seed=mission_seed, tier=self.selected_tier)
            apply_human_timer_assist(
                self.sim, bool(self.settings["accessibility"].get("timer_assist", False))
            )
            self.agent_env = None
            self.agent_hidden = None
        self.renderer.reset_for_sim(self.sim)
        self._agent_tick = 0
        self._agent_action = self.policy.act(self.sim)
        self._agent_latency_ms = 0.0
        self._reset_telemetry(agent=agent)
        self.state = "lab_play" if agent else "play"

    def _reset_telemetry(self, *, agent: bool) -> None:
        self._telemetry = {
            "schema_version": 1,
            "controller": "agent" if agent else "human",
            "policy": self.policy_name if agent else "keyboard",
            "seed": self.sim.seed,
            "tier": self.sim.tier,
            "timer_assistance": bool(
                not agent and self.settings["accessibility"].get("timer_assist", False)
            ),
            "success": False,
            "actions": 0,
            "action_counts": [0] * 36,
            "policy_decisions": 0,
            "idle_ticks": 0,
            "path": [[0.0, round(float(self.sim.player[0]), 2), round(float(self.sim.player[1]), 2)]],
            "trace_curve": [[0.0, 0.0]],
            "policy_latencies_ms": [],
            "last_sample_second": 0,
            "recorded": False,
            "ideal_distance": self._ideal_geometric_distance(),
        }

    def _ideal_geometric_distance(self) -> float:
        terminals = self.sim.level.terminals
        best = math.inf
        for count in range(1, len(terminals) + 1):
            for subset in itertools.permutations(terminals, count):
                if sum(terminal.value for terminal in subset) < self.sim.level.quota:
                    continue
                points = [self.sim.level.spawn, *(terminal.position for terminal in subset), self.sim.level.extraction]
                distance = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
                best = min(best, distance)
            if math.isfinite(best):
                break
        return 0.0 if not math.isfinite(best) else best

    @staticmethod
    def _touch_circle_hit(point: tuple[float, float], center: tuple[int, int], radius: int) -> bool:
        return math.hypot(point[0] - center[0], point[1] - center[1]) <= radius

    def _touch_logical_position(self, event: pygame.event.Event) -> tuple[float, float] | None:
        window_position = self._event_window_position(event)
        return self.renderer.logical_point(window_position) if window_position is not None else None

    def _handle_touch_events(self, events: list[pygame.event.Event]) -> bool:
        """Update multi-touch roles and report a pause-button press."""

        for event in events:
            is_finger = event.type in (pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP)
            is_mouse = event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP)
            if not is_finger and not (is_mouse and self.touch_controls_enabled):
                continue
            if is_mouse and bool(getattr(event, "touch", False)):
                # SDL emits a matching finger event for synthesized touch mice.
                continue
            if is_finger:
                self.touch_controls_enabled = True
                contact = int(event.finger_id)
                phase = {
                    pygame.FINGERDOWN: "down",
                    pygame.FINGERMOTION: "move",
                    pygame.FINGERUP: "up",
                }[event.type]
            else:
                contact = -1
                if event.type == pygame.MOUSEBUTTONDOWN and int(getattr(event, "button", 0)) == 1:
                    phase = "down"
                elif event.type == pygame.MOUSEBUTTONUP and int(getattr(event, "button", 0)) == 1:
                    phase = "up"
                elif event.type == pygame.MOUSEMOTION and contact in self._touch_roles:
                    phase = "move"
                else:
                    continue

            if phase == "up":
                self._touch_points.pop(contact, None)
                self._touch_roles.pop(contact, None)
                continue

            point = self._touch_logical_position(event)
            if point is None:
                continue
            if phase == "down":
                if TOUCH_PAUSE_RECT.collidepoint(point):
                    return True
                if self._touch_circle_hit(point, TOUCH_DASH_CENTER, TOUCH_DASH_RADIUS + 8):
                    role = "dash"
                elif self._touch_circle_hit(point, TOUCH_PULSE_CENTER, TOUCH_PULSE_RADIUS + 8):
                    role = "pulse"
                elif point[0] <= 180 and point[1] >= 210:
                    role = "move"
                else:
                    continue
                self._touch_roles[contact] = role
            if contact in self._touch_roles:
                self._touch_points[contact] = point
        return False

    def _touch_action(self) -> Action:
        move_point = next(
            (self._touch_points[contact] for contact, role in self._touch_roles.items() if role == "move"),
            None,
        )
        move = 0
        if move_point is not None:
            dx = move_point[0] - TOUCH_JOYSTICK_CENTER[0]
            dy = move_point[1] - TOUCH_JOYSTICK_CENTER[1]
            if math.hypot(dx, dy) >= 9.0:
                sector = int(round(math.atan2(dy, dx) / (math.pi / 4.0))) % 8
                move = (3, 4, 5, 6, 7, 8, 1, 2)[sector]
        roles = set(self._touch_roles.values())
        return Action(move=move, dash="dash" in roles, pulse="pulse" in roles)

    def _touch_visual_state(self) -> dict[str, Any]:
        move_point = next(
            (self._touch_points[contact] for contact, role in self._touch_roles.items() if role == "move"),
            None,
        )
        roles = set(self._touch_roles.values())
        return {"move_point": move_point, "dash": "dash" in roles, "pulse": "pulse" in roles}

    def _play(self, *, agent: bool) -> None:
        events = self._events()
        if not agent and self._handle_touch_events(events):
            self._touch_points.clear()
            self._touch_roles.clear()
            self.state, self.selection = "pause", 0
            return
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == self._key("pause"):
                if agent:
                    self.state = "lab"
                    return
                self.state, self.selection = "pause", 0
                return
            if event.key == self._key("restart"):
                self._record_current_run(abandoned=True)
                self._start_mission(agent=agent, replay_seed=self.sim.seed)
                return
        if not self.running or self.state not in ("play", "lab_play"):
            return

        if agent:
            if self._agent_tick % 6 == 0:
                started = time.perf_counter()
                if self.learned_policy is not None:
                    self.agent_observation = self.agent_env._observation()
                    self._agent_action, self.agent_hidden = self.learned_policy.act(
                        self.agent_observation, self.agent_hidden, deterministic=True
                    )
                else:
                    self._agent_action = self.policy.act(self.sim)
                measured = (time.perf_counter() - started) * 1000.0
                self._agent_latency_ms = float(getattr(self.learned_policy, "last_latency_ms", measured)) if self.learned_policy is not None else measured
                if bool(getattr(self.learned_policy, "waiting_for_action", False)):
                    # Browser inference is asynchronous. Preserve exact
                    # checkpoint behavior by holding simulation time at the
                    # policy boundary instead of inventing a neutral action or
                    # prefetching from a stale mid-repeat state.
                    self.renderer.draw(
                        lab_stats={
                            "policy": self.policy_name,
                            "action": "SYNCING EXACT STATE",
                            "latency_ms": self._agent_latency_ms,
                            "hidden": "--",
                        },
                        compact_hud=self.touch_controls_enabled,
                    )
                    return
                self._telemetry["policy_decisions"] += 1
                self._telemetry["policy_latencies_ms"].append(round(self._agent_latency_ms, 4))
            action = Action.decode(self._agent_action)
            self._agent_tick += 1
        else:
            keys = pygame.key.get_pressed()
            vertical = int(keys[self._key("move_down")]) - int(keys[self._key("move_up")])
            horizontal = int(keys[self._key("move_right")]) - int(keys[self._key("move_left")])
            move_lookup = {
                (0, 0): 0,
                (0, -1): 1,
                (1, -1): 2,
                (1, 0): 3,
                (1, 1): 4,
                (0, 1): 5,
                (-1, 1): 6,
                (-1, 0): 7,
                (-1, -1): 8,
            }
            touch_action = self._touch_action()
            keyboard_move = move_lookup[(horizontal, vertical)]
            action = Action(
                keyboard_move or touch_action.move,
                bool(keys[self._key("dash")]) or touch_action.dash,
                bool(keys[self._key("pulse")]) or touch_action.pulse,
            )
        self._telemetry["actions"] += 1
        self._telemetry["action_counts"][action.encode()] += 1
        self._telemetry["idle_ticks"] += int(action.move == 0)
        self.sim.advance(action, ticks=1)
        self._sample_telemetry()
        sim_events = self.sim.pop_events()
        self.audio.handle(sim_events)
        self.audio.update(trace=self.sim.trace, lockdown=self.sim.lockdown, speed=float(np.linalg.norm(self.sim.velocity)))
        self.renderer.ingest_events(sim_events)
        hidden_text = "--"
        if self.agent_hidden is not None:
            hidden = self.agent_hidden.detach().cpu().numpy() if hasattr(self.agent_hidden, "detach") else np.asarray(self.agent_hidden)
            hidden_text = f"{float(np.linalg.norm(hidden)):.1f}"
        decoded = Action.decode(self._agent_action) if agent else action
        action_text = MOVE_LABELS[decoded.move] + (" +DASH" if decoded.dash else "") + (" +PULSE" if decoded.pulse else "")
        self.renderer.draw(
            lab_stats={
                "policy": self.policy_name,
                "action": action_text,
                "latency_ms": self._agent_latency_ms,
                "hidden": hidden_text,
            }
            if agent
            else None,
            touch_controls=self._touch_visual_state() if self.touch_controls_enabled and not agent else None,
            compact_hud=self.touch_controls_enabled,
        )
        if self.sim.terminated or self.sim.truncated:
            self._debrief_agent = agent
            self._record_current_run()
            self.state = "debrief"
            if self.sim.extracted and not agent:
                record_success(tier=self.selected_tier, score=mission_score(self.sim))
            self.progression = load_progression()

    def _sample_telemetry(self) -> None:
        second = int(self.sim.elapsed_seconds)
        if second <= int(self._telemetry["last_sample_second"]):
            return
        self._telemetry["last_sample_second"] = second
        self._telemetry["path"].append([round(self.sim.elapsed_seconds, 2), round(float(self.sim.player[0]), 2), round(float(self.sim.player[1]), 2)])
        self._telemetry["trace_curve"].append([round(self.sim.elapsed_seconds, 2), round(float(self.sim.trace), 3)])

    def _record_current_run(self, *, abandoned: bool = False) -> None:
        if not self._telemetry or self._telemetry.get("recorded"):
            return
        self._sample_telemetry()
        distance = float(self.sim.distance_travelled)
        ideal = float(self._telemetry.pop("ideal_distance", 0.0))
        latencies = list(self._telemetry.pop("policy_latencies_ms", []))
        actions = max(1, int(self._telemetry.get("actions", 1)))
        self._telemetry.update(
            {
                "success": bool(self.sim.extracted),
                "failure_reason": "abandoned" if abandoned else self.sim.fail_reason,
                "duration_seconds": round(self.sim.elapsed_seconds, 3),
                "quota": self.sim.level.quota,
                "data": self.sim.data,
                "optional_data": self.sim.optional_data,
                "max_trace": round(float(self.sim.max_trace), 3),
                "mean_trace": round(float(np.mean([point[1] for point in self._telemetry["trace_curve"]])), 3),
                "detections": self.sim.detections,
                "damage": self.sim.damage_taken,
                "pulse_usage": self.sim.pulses_used,
                "distance_travelled": round(distance, 3),
                "path_efficiency": round(min(1.0, ideal / max(ideal, distance, 1e-6)), 4),
                "idle_fraction": round(float(self._telemetry["idle_ticks"]) / actions, 4),
                "idle_decisions": int(self._telemetry["idle_ticks"]),
                "decision_count": int(self._telemetry["actions"]),
                "policy_latency_mean_ms": round(float(np.mean(latencies)), 4) if latencies else 0.0,
                "policy_latency_p95_ms": round(float(np.percentile(latencies, 95)), 4) if latencies else 0.0,
            }
        )
        self._telemetry.pop("last_sample_second", None)
        self._telemetry["recorded"] = True
        record_run({key: value for key, value in self._telemetry.items() if key != "recorded"})

    def _pause(self) -> None:
        items = ["RESUME", "RESTART CONTRACT", "ACCESSIBILITY", "ABANDON TO MENU"]
        choice = self._menu_events(self._events(), len(items))
        if choice == "back" or (choice == "confirm" and self.selection == 0):
            self.state = "play"
        elif choice == "confirm" and self.selection == 1:
            self._record_current_run(abandoned=True)
            self._start_mission(agent=False, replay_seed=self.sim.seed)
        elif choice == "confirm" and self.selection == 2:
            self._settings_return_state = "pause"
            self.state, self.selection = "settings_accessibility", 0
        elif choice == "confirm" and self.selection == 3:
            self._record_current_run(abandoned=True)
            self.state, self.selection = "main", 0
        self.renderer.draw_screen(
            title="PAUSED",
            subtitle=f"TIER {self.selected_tier} // SEED {self.sim.seed}",
            items=items,
            selected=self.selection,
            panel=[
                f"DATA       {self.sim.data}/{self.sim.level.quota}",
                f"TRACE      {self.sim.trace:04.1f}",
                f"INTEGRITY  {self.sim.integrity}/3",
                f"TIME       {self.sim.remaining_seconds:05.1f}s",
                "",
                "Changes are saved immediately.",
            ],
            footer=f"{self._key_label('back')}  RESUME",
        )

    def _debrief(self) -> None:
        events = self._events()
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == self._key("confirm"):
                if self._debrief_agent:
                    self._start_mission(agent=True, replay_seed=self.lab_seed + 1)
                    self.lab_seed += 1
                elif not self.sim.extracted:
                    self._start_mission(agent=False, replay_seed=self.sim.seed)
                else:
                    self.state, self.selection = "stage_select", self.selected_tier - 1
            elif event.type == pygame.KEYDOWN and event.key == self._key("back"):
                self.state, self.selection = "main", 0
        status = "CONTRACT CLEARED" if self.sim.extracted else "CONTRACT FAILED"
        score = mission_score(self.sim)
        path_efficiency = float(self._telemetry.get("path_efficiency", 0.0))
        badges = []
        if self.sim.extracted:
            if self.sim.detections == 0:
                badges.append("GHOST")
            if self.sim.damage_taken == 0:
                badges.append("NO DAMAGE")
            if self.sim.optional_data > 0:
                badges.append("OPTIONAL DATA")
            if path_efficiency >= 0.80:
                badges.append("EFFICIENT ROUTE")
        outcome_lines = [f"BADGES      {' // '.join(badges[:2]) if badges else 'NONE'}"]
        if not self.sim.extracted:
            outcome_lines = [f"FAILURE     {self.sim.fail_reason.replace('_', ' ').upper()}"]
        elif len(badges) > 2:
            # Keep earned badge names atomic.  Letting the generic menu wrapper
            # split this sentence could render ``EFFICIENT`` and ``ROUTE`` on
            # separate lines depending on which other badges were earned.
            outcome_lines.append(" // ".join(badges[2:]))
        body = [
            f"DATA        {self.sim.data}/{self.sim.level.quota}  (+{self.sim.optional_data} OPTIONAL)",
            f"TIME        {self.sim.elapsed_seconds:6.1f}s",
            f"MAX TRACE   {self.sim.max_trace:6.1f}%",
            f"DETECTIONS  {self.sim.detections}",
            f"INTEGRITY   {self.sim.integrity}/3",
            f"EFFICIENCY  {path_efficiency * 100:5.1f}%",
            *outcome_lines,
            f"SCORE       {score:06d}",
        ]
        if self._debrief_agent:
            footer = f"{self._key_label('confirm')}  NEXT SEED     {self._key_label('back')}  MAIN MENU"
        elif self.sim.extracted:
            footer = f"{self._key_label('confirm')}  CONTRACTS     {self._key_label('back')}  MAIN MENU"
        else:
            footer = f"{self._key_label('confirm')}  RETRY SAME SEED     {self._key_label('back')}  MAIN MENU"
        self.renderer.draw_screen(
            title=status,
            subtitle=f"TIER {self.selected_tier} // SEED {self.sim.seed}",
            body=body,
            badge="AGENT RUN" if self._debrief_agent else "OPERATIVE RUN",
            panel=[
                "BENCHMARK RECORD",
                f"CONTROLLER  {'AGENT' if self._debrief_agent else 'HUMAN'}",
                f"DISTANCE    {self.sim.distance_travelled:7.1f}px",
                f"IDLE        {float(self._telemetry.get('idle_fraction', 0)) * 100:5.1f}%",
                f"PULSE USE   {self.sim.pulses_used}",
                f"POLICY P95  {float(self._telemetry.get('policy_latency_p95_ms', 0)):5.2f}ms" if self._debrief_agent else "POLICY P95  —",
            ],
            footer=footer,
        )

    def _how_to(self) -> None:
        for event in self._events():
            if event.type == pygame.KEYDOWN and event.key in (self._key("back"), self._key("confirm")):
                self.state, self.selection = "main", 0
        self.renderer.draw_screen(
            title="FIELD MANUAL",
            subtitle="A contract is information, pressure, and an exit.",
            body=[
                f"{self._key_label('move_up')}/{self._key_label('move_left')}/{self._key_label('move_down')}/{self._key_label('move_right')}   MOVE",
                f"{self._key_label('dash'):<11} DASH // FAST, LOUD, ENERGY-LIMITED",
                f"{self._key_label('pulse'):<11} DISRUPTION PULSE // LIMITED CHARGES",
                "",
                "Enter an amber ring to link. Movement inside never interrupts it.",
                "Leaving pauses the link; returning resumes it at the same progress.",
                "Break sight to cool TRACE. Pink/red shapes mean immediate danger.",
                "Meet quota, then reach the blue/green extraction relay.",
            ],
            compact_body=True,
            panel=[
                "READ THE SECURITY LAYER",
                "Faint amber marks the real sight envelope.",
                "Square + dashed beam: camera.",
                "Triangle + edge notches: guard.",
                "I / II / III: Standard / Interceptor / Elite patrol.",
                "Dim LAST markers freeze at the last position you actually saw.",
                "Segments fill before detection; walls reset pressure.",
                "Pulse disables electronics; dash makes noise.",
            ],
            footer=f"{self._key_label('back')} OR {self._key_label('confirm')}  BACK",
        )

    def _settings(self) -> None:
        items = ["AUDIO MIX", "ACCESSIBILITY", "CONTROLS", "DISPLAY", "BACK"]
        choice = self._menu_events(self._events(), len(items))
        if choice == "back" or (choice == "confirm" and self.selection == 4):
            self.state, self.selection = "main", 0
        elif choice == "confirm":
            self._settings_return_state = "settings"
            self.state = ("settings_audio", "settings_accessibility", "settings_controls", "settings_display")[self.selection]
            self.selection = 0
        self.renderer.draw_screen(
            title="SETTINGS",
            subtitle="Saved instantly to your local Ghostline profile.",
            items=items,
            selected=self.selection,
            panel=[
                "ACCESS PROFILE",
                f"CAPTIONS       {self._on_off(self.settings['accessibility']['sound_captions'])}",
                f"HIGH CONTRAST  {self._on_off(self.settings['accessibility']['high_contrast'])}",
                f"COLOR-SAFE     {self._on_off(self.settings['accessibility']['color_safe'])}",
                f"HUD SCALE      {int(self.settings['accessibility']['hud_scale'] * 100)}%",
                "",
                "Every gameplay key is remappable.",
            ],
            footer=f"{self._key_label('confirm')}  OPEN     {self._key_label('back')}  BACK",
        )

    @staticmethod
    def _on_off(value: object) -> str:
        return "ON" if bool(value) else "OFF"

    def _cycle_volume(self, key: str) -> None:
        value = float(self.settings["audio"][key])
        self.settings["audio"][key] = round((value + 0.1) % 1.1, 2)
        self._persist_settings()

    def _settings_audio(self) -> None:
        audio = self.settings["audio"]
        items = [
            f"AUDIO          {self._on_off(audio['enabled'])}",
            f"MASTER         {int(audio['master'] * 100):3d}%",
            f"MUSIC          {int(audio['music'] * 100):3d}%",
            f"SOUND EFFECTS  {int(audio['sfx'] * 100):3d}%",
            "BACK",
        ]
        choice = self._menu_events(self._events(), len(items))
        if choice == "back" or (choice == "confirm" and self.selection == 4):
            self.state, self.selection = "settings", 0
        elif choice == "confirm" and self.selection == 0:
            audio["enabled"] = not audio["enabled"]
            self._persist_settings()
        elif choice == "confirm" and self.selection in (1, 2, 3):
            self._cycle_volume(("master", "music", "sfx")[self.selection - 1])
        self.renderer.draw_screen(
            title="AUDIO MIX",
            subtitle="Procedural ambience responds to network pressure.",
            items=items,
            selected=self.selection,
            panel=["MASTER controls the full mix.", "MUSIC controls ambience + tension.", "SFX controls gameplay and UI cues.", "", "Sound captions are under Accessibility."],
            footer=f"{self._key_label('confirm')}  TOGGLE / +10%     {self._key_label('back')}  BACK",
        )

    def _settings_accessibility(self) -> None:
        access = self.settings["accessibility"]
        items = [
            f"HIGH CONTRAST   {self._on_off(access['high_contrast'])}",
            f"COLOR-SAFE CUES {self._on_off(access['color_safe'])}",
            f"REDUCED MOTION  {self._on_off(access['reduced_motion'])}",
            f"REDUCED FLASHES {self._on_off(access['reduced_flashes'])}",
            f"SOUND CAPTIONS  {self._on_off(access['sound_captions'])}",
            f"HUD SCALE       {int(access['hud_scale'] * 100)}%",
            f"TIMER ASSIST    {self._on_off(access['timer_assist'])}",
            f"TIMER WARNINGS  {self._on_off(access['timer_warnings'])}",
            f"TUTORIAL HINTS  {self._on_off(access['tutorial_hints'])}",
            "BACK",
        ]
        choice = self._menu_events(self._events(), len(items), compact=True)
        if choice == "back" or (choice == "confirm" and self.selection == 9):
            target = self._settings_return_state
            self.state, self.selection = target, (0 if target == "pause" else 1)
        elif choice == "confirm":
            keys = ("high_contrast", "color_safe", "reduced_motion", "reduced_flashes", "sound_captions")
            if self.selection < len(keys):
                key = keys[self.selection]
                access[key] = not access[key]
            elif self.selection == 5:
                scales = (1.0, 1.25, 1.5)
                access["hud_scale"] = scales[(scales.index(float(access["hud_scale"])) + 1) % len(scales)]
            elif self.selection == 6:
                access["timer_assist"] = not access["timer_assist"]
            elif self.selection == 7:
                access["timer_warnings"] = not access["timer_warnings"]
            elif self.selection == 8:
                access["tutorial_hints"] = not access["tutorial_hints"]
            self._persist_settings()
        self.renderer.draw_screen(
            title="ACCESSIBILITY",
            subtitle="Danger always uses color plus shape and text.",
            items=items,
            selected=self.selection,
            compact=True,
            panel=["TIMER ASSIST adds 35% to human contract windows and is recorded in telemetry.", "REDUCED MOTION disables shake, trails, and moving UI art.", "COLOR-SAFE remaps danger to pink and extraction to blue.", "CAPTIONS identify alerts, impacts, pulses, and terminals."],
            footer=f"{self._key_label('confirm')}  TOGGLE     {self._key_label('back')}  BACK",
        )

    def _settings_display(self) -> None:
        display = self.settings["display"]
        items = [
            f"FULLSCREEN     {self._on_off(display['fullscreen'])}",
            f"SCREEN SHAKE   {self._on_off(display['screen_shake'])}",
            "BACK",
        ]
        choice = self._menu_events(self._events(), len(items))
        if choice == "back" or (choice == "confirm" and self.selection == 2):
            self.state, self.selection = "settings", 3
        elif choice == "confirm" and self.selection == 0:
            display["fullscreen"] = not display["fullscreen"]
            self.fullscreen = bool(display["fullscreen"])
            flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
            self.renderer.window = pygame.display.set_mode((0, 0) if self.fullscreen else (1280, 720), flags)
            self._persist_settings()
        elif choice == "confirm" and self.selection == 1:
            display["screen_shake"] = not display["screen_shake"]
            self._persist_settings()
        self.renderer.draw_screen(
            title="DISPLAY",
            subtitle="Pixel-precise world // native-resolution interface.",
            items=items,
            selected=self.selection,
            panel=["The world keeps its authored pixel grid.", "Text is redrawn at the window's real resolution.", "16:9 sizes fill cleanly; other ratios letterbox.", "", "Reduced Motion always suppresses shake."],
            footer=f"{self._key_label('confirm')}  TOGGLE     {self._key_label('back')}  BACK",
        )

    def _settings_controls(self) -> None:
        events = self._events()
        if self._rebind_action is not None:
            for event in events:
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE and self._rebind_action not in ("pause", "back"):
                    self._rebind_action = None
                    break
                self._assign_binding(self._rebind_action, pygame.key.name(event.key))
                self._rebind_action = None
                self.audio.menu_confirm()
                break
        else:
            all_items = [f"{label:<15} {self._key_label(action)}" for action, label in CONTROL_ACTIONS]
            all_items += ["RESTORE DEFAULTS", "BACK"]
            window_size = 10
            window_start = max(0, min(self.selection - window_size // 2, len(all_items) - window_size))
            choice = self._menu_events(
                events,
                len(all_items),
                compact=True,
                pointer_count=window_size,
                pointer_offset=window_start,
            )
            if choice == "back" or (choice == "confirm" and self.selection == len(CONTROL_ACTIONS) + 1):
                self.state, self.selection = "settings", 2
            elif choice == "confirm" and self.selection == len(CONTROL_ACTIONS):
                self.settings["bindings"] = dict(DEFAULT_BINDINGS)
                self._persist_settings()
            elif choice == "confirm":
                self._rebind_action = CONTROL_ACTIONS[self.selection][0]
        all_items = [f"{label:<15} {self._key_label(action)}" for action, label in CONTROL_ACTIONS]
        all_items += ["RESTORE DEFAULTS", "BACK"]
        window_size = 10
        window_start = max(0, min(self.selection - window_size // 2, len(all_items) - window_size))
        items = all_items[window_start : window_start + window_size]
        visible_selection = self.selection - window_start
        subtitle = "Press a new key now // Escape cancels" if self._rebind_action else "Select an action, then press any keyboard key."
        self.renderer.draw_screen(
            title="CONTROLS",
            subtitle=subtitle,
            items=items,
            selected=visible_selection,
            compact=True,
            panel=["Duplicate keys are resolved by swapping bindings.", "Arrow keys remain emergency menu navigation.", "", f"ITEM       {self.selection + 1}/{len(all_items)}", f"LISTENING  {self._rebind_action.upper() if self._rebind_action else 'NO'}"],
            footer="PRESS KEY" if self._rebind_action else f"{self._key_label('confirm')}  REBIND     {self._key_label('back')}  BACK",
        )

    def _assign_binding(self, action: str, name: str) -> None:
        name = name.strip().lower()
        bindings = self.settings["bindings"]
        previous = str(bindings[action])
        gameplay_actions = {"move_up", "move_down", "move_left", "move_right", "dash", "pulse", "restart", "pause"}
        menu_actions = {"menu_up", "menu_down", "confirm", "back"}
        group = gameplay_actions if action in gameplay_actions else menu_actions
        conflict = next((other for other, current in bindings.items() if other in group and other != action and current == name), None)
        bindings[action] = name
        if conflict is not None:
            bindings[conflict] = previous
        self._persist_settings()

    def _persist_settings(self) -> None:
        self.settings = save_settings(self.settings)
        self.renderer.screen_shake_enabled = bool(self.settings["display"]["screen_shake"])
        self.renderer.apply_accessibility(self.settings["accessibility"])
        self._apply_audio_settings()
        self.progression = load_progression()

    def _apply_audio_settings(self) -> None:
        audio = self.settings["audio"]
        self.audio.set_enabled(bool(audio["enabled"]))
        self.audio.set_mix(master=float(audio["master"]), music=float(audio["music"]), sfx=float(audio["sfx"]))

    def _credits(self) -> None:
        for event in self._events():
            if event.type == pygame.KEYDOWN and event.key in (self._key("back"), self._key("confirm")):
                self.state, self.selection = "main", 0
        self.renderer.draw_screen(
            title="CREDITS",
            subtitle="Ghostline // procedural infiltration and RL benchmark",
            body=[
                "DESIGN, ENGINEERING, SIMULATION  //  PROJECT AUTHOR + CODEX",
                "VISUAL SYSTEM  //  ORIGINAL PROCEDURAL PIXEL ART + MANUAL QA",
                "AUDIO  //  ORIGINAL RUNTIME SYNTHESIS",
                "RL STACK  //  GYMNASIUM + TORCHRL + ONNX RUNTIME",
                "",
                "One deterministic world serves human and policy controllers.",
                "See assets/licenses.json and the project wiki for disclosure.",
            ],
            compact_body=True,
            panel=["PORTFOLIO PRINCIPLES", "Held-out procedural evaluation", "Player-equivalent policy inputs", "Exact reward accounting", "Deterministic replay", "Transparent AI assistance"],
            footer=f"{self._key_label('back')} OR {self._key_label('confirm')}  BACK",
        )
