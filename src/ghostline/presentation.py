from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Iterable

import numpy as np
import pygame

from ghostline.config import (
    CAMERA_VISION_COSINE,
    CAMERA_VISION_DISTANCE,
    DRONE_STRIKE_WINDUP_SECONDS,
    GUARD_VISION_BASE_DISTANCE,
    GUARD_VISION_COSINE,
    GUARD_VISION_DISTANCE_PER_ALERT,
    HACK_RADIUS,
    PLAYER_PERCEPTION_DISTANCE,
    GUARD_STRIKE_WINDUP_SECONDS,
    ROLE_COLORS,
    TILE_SIZE,
    TIERS,
    TRACE_MAX,
)
from ghostline.simulation import GhostlineSimulation, angle_vector, norm
from ghostline.resources import runtime_asset_path
from ghostline.types import GuardGrade, GuardMode, Prop, SimEvent, Tile

LOGICAL_SIZE = (640, 360)
WINDOW_SIZE = (1280, 720)


def _presentation_scaled_size(
    target_size: tuple[int, int],
    *,
    web_runtime: bool | None = None,
) -> tuple[int, int]:
    """Choose the output size for desktop and browser presentation.

    Desktop keeps crisp integer scaling. Browser canvas dimensions follow CSS
    pixels and device zoom, so an intermediate 16:9 size such as 927x521 must
    fill the canvas instead of falling back to a centered 640x360 image.
    """

    if web_runtime is None:
        web_runtime = sys.platform == "emscripten"
    if web_runtime:
        return max(1, int(target_size[0])), max(1, int(target_size[1]))
    integer_scale = max(
        1,
        min(target_size[0] // LOGICAL_SIZE[0], target_size[1] // LOGICAL_SIZE[1]),
    )
    if target_size[0] < LOGICAL_SIZE[0] or target_size[1] < LOGICAL_SIZE[1]:
        scale = min(
            target_size[0] / LOGICAL_SIZE[0],
            target_size[1] / LOGICAL_SIZE[1],
        )
        return (
            max(1, int(LOGICAL_SIZE[0] * scale)),
            max(1, int(LOGICAL_SIZE[1] * scale)),
        )
    return LOGICAL_SIZE[0] * integer_scale, LOGICAL_SIZE[1] * integer_scale

BG = (8, 12, 17)
INK = (222, 235, 232)
MUTED = (121, 142, 145)
CYAN = (72, 231, 218)
TEAL = (44, 165, 157)
AMBER = (245, 184, 76)
RED = (244, 78, 88)
VIOLET = (172, 101, 255)
GREEN = (87, 226, 139)

# Presentation mirrors the simulation's literal sight contract.  Keeping these
# values named here makes it difficult for the visible warning footprint to
# silently drift from the real camera/guard envelope again.
CAMERA_VIEW_DISTANCE = CAMERA_VISION_DISTANCE
CAMERA_VIEW_HALF_ANGLE = math.acos(CAMERA_VISION_COSINE)
GUARD_VIEW_BASE_DISTANCE = GUARD_VISION_BASE_DISTANCE
GUARD_VIEW_ALERT_DISTANCE = GUARD_VISION_DISTANCE_PER_ALERT
GUARD_VIEW_HALF_ANGLE = math.acos(GUARD_VISION_COSINE)


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    color: tuple[int, int, int]
    life: float
    size: int


@dataclass
class Caption:
    text: str
    life: float
    color: tuple[int, int, int]


@dataclass
class ScreenEffect:
    kind: str
    position: np.ndarray
    life: float
    duration: float


@dataclass(frozen=True)
class NativeTextCommand:
    """One glyph run redrawn at the actual output resolution.

    The world deliberately remains a 640x360 pixel-art canvas. Text does not:
    rasterizing glyphs into that canvas made a 1080p/fullscreen window enlarge
    the same low-resolution pixels. Keeping logical coordinates while drawing
    the final glyph run after world scaling preserves the exact field of view
    and gives every HUD, menu, caption, and Agent Lab label native-resolution
    edges.
    """

    text: str
    x: float
    y: float
    design_size: int
    color: tuple[int, int, int]


@dataclass(frozen=True)
class AtlasRegion:
    rect: tuple[int, int, int, int]
    maximum_world_width: int


@dataclass(frozen=True)
class CharacterRegion:
    rect: tuple[int, int, int, int]


# Alpha-clean release-atlas coordinates. Crops include two pixels of transparent
# padding so nearest-neighbour scaling cannot clip edge pixels.
ATLAS_REGIONS: dict[str, AtlasRegion] = {
    "desk": AtlasRegion((26, 444, 76, 72), 64),
    "console": AtlasRegion((120, 442, 74, 73), 72),
    "chair": AtlasRegion((210, 437, 39, 73), 32),
    "plant": AtlasRegion((322, 432, 47, 79), 32),
    "sofa": AtlasRegion((436, 442, 124, 67), 96),
    "lab_bench": AtlasRegion((27, 538, 77, 79), 72),
    "server": AtlasRegion((660, 532, 67, 86), 52),
    "locker": AtlasRegion((854, 447, 36, 61), 32),
    "camera": AtlasRegion((1362, 451, 55, 41), 26),
    "terminal": AtlasRegion((1338, 371, 77, 49), 30),
    "vault_case": AtlasRegion((26, 649, 86, 90), 96),
    "crate": AtlasRegion((657, 659, 62, 81), 64),
    "generator": AtlasRegion((135, 772, 85, 101), 72),
}


CHARACTER_REGIONS: dict[str, CharacterRegion] = {
    "runner_s": CharacterRegion((45, 68, 71, 129)),
    "runner_se": CharacterRegion((167, 68, 70, 132)),
    "runner_e": CharacterRegion((289, 68, 56, 132)),
    "runner_n": CharacterRegion((514, 68, 69, 132)),
    "runner_ne": CharacterRegion((628, 68, 71, 132)),
    "runner_run_a": CharacterRegion((748, 73, 82, 118)),
    "runner_run_b": CharacterRegion((865, 74, 111, 117)),
    "runner_run_c": CharacterRegion((1004, 73, 96, 118)),
    "runner_dash": CharacterRegion((1138, 94, 167, 103)),
    "runner_link": CharacterRegion((1379, 70, 133, 131)),
    "runner_damage": CharacterRegion((1505, 91, 130, 112)),
    "guard_s": CharacterRegion((41, 283, 77, 148)),
    "guard_se": CharacterRegion((165, 283, 74, 151)),
    "guard_e": CharacterRegion((285, 283, 60, 151)),
    "guard_n": CharacterRegion((514, 283, 74, 148)),
    "guard_ne": CharacterRegion((634, 283, 74, 150)),
    "guard_walk": CharacterRegion((763, 287, 89, 145)),
    "guard_chase": CharacterRegion((903, 287, 100, 143)),
    "guard_suspicious": CharacterRegion((1086, 287, 86, 147)),
    "guard_run": CharacterRegion((1242, 304, 102, 130)),
    "guard_strike": CharacterRegion((1413, 310, 153, 122)),
    "drone_s": CharacterRegion((41, 536, 94, 77)),
    "drone_se": CharacterRegion((187, 536, 84, 77)),
    "drone_e": CharacterRegion((324, 537, 81, 77)),
    "drone_n": CharacterRegion((575, 541, 88, 73)),
    "drone_ne": CharacterRegion((715, 539, 80, 79)),
    "drone_flight": CharacterRegion((1004, 524, 89, 101)),
    "drone_charge": CharacterRegion((1316, 495, 119, 135)),
    "drone_recoil": CharacterRegion((1523, 511, 105, 113)),
    "camera_active": CharacterRegion((191, 726, 97, 107)),
    "camera_detected": CharacterRegion((500, 730, 77, 107)),
    "camera_suspicious": CharacterRegion((797, 730, 95, 102)),
    "camera_disabled": CharacterRegion((1245, 731, 93, 103)),
    "camera_damaged": CharacterRegion((1532, 732, 101, 123)),
}

# Dedicated direction-preserving diagonal locomotion loops.  Each row in the
# reviewed alpha atlas is one four-frame cycle with independently cleaned crops.
DIAGONAL_LOCOMOTION_REGIONS: dict[str, tuple[CharacterRegion, ...]] = {
    "runner_ne": tuple(
        CharacterRegion(rect)
        for rect in (
            (120, 51, 140, 199),
            (394, 56, 165, 185),
            (688, 56, 176, 189),
            (996, 51, 162, 189),
        )
    ),
    "runner_se": tuple(
        CharacterRegion(rect)
        for rect in (
            (103, 354, 148, 193),
            (399, 360, 148, 175),
            (670, 357, 175, 188),
            (956, 356, 174, 181),
        )
    ),
    "guard_ne": tuple(
        CharacterRegion(rect)
        for rect in (
            (120, 655, 140, 199),
            (394, 661, 165, 186),
            (688, 661, 176, 188),
            (996, 659, 162, 189),
        )
    ),
    "guard_se": tuple(
        CharacterRegion(rect)
        for rect in (
            (103, 961, 148, 192),
            (399, 966, 149, 175),
            (670, 962, 175, 188),
            (956, 963, 174, 181),
        )
    ),
}

RUNNER_DIRECTION_REGIONS: dict[int, tuple[str, bool]] = {
    0: ("runner_e", False),
    1: ("runner_se", False),
    2: ("runner_s", False),
    3: ("runner_se", True),
    4: ("runner_e", True),
    5: ("runner_ne", True),
    6: ("runner_n", False),
    7: ("runner_ne", False),
}

GUARD_DIRECTION_REGIONS: dict[int, tuple[str, bool]] = {
    0: ("guard_e", False),
    1: ("guard_se", False),
    2: ("guard_s", False),
    3: ("guard_se", True),
    4: ("guard_e", True),
    5: ("guard_ne", True),
    6: ("guard_n", False),
    7: ("guard_ne", False),
}

DRONE_DIRECTION_REGIONS: dict[int, tuple[str, bool]] = {
    0: ("drone_e", False),
    1: ("drone_se", False),
    2: ("drone_s", False),
    3: ("drone_se", True),
    4: ("drone_e", True),
    5: ("drone_ne", True),
    6: ("drone_n", False),
    7: ("drone_ne", False),
}


class GhostlineRenderer:
    """Pixel-art scrolling presentation consuming simulation state and events."""

    def __init__(
        self,
        sim: GhostlineSimulation,
        *,
        visible: bool = True,
        screen_shake: bool = True,
        accessibility: dict[str, Any] | None = None,
    ):
        pygame.init()
        pygame.font.init()
        self.sim = sim
        self.visible = visible
        self.screen_shake_enabled = screen_shake
        flags = pygame.RESIZABLE if visible else pygame.HIDDEN
        self.window = pygame.display.set_mode(WINDOW_SIZE, flags)
        pygame.display.set_caption("GHOSTLINE // PROCEDURAL INFILTRATION")
        self.logical = pygame.Surface(LOGICAL_SIZE).convert()
        self.light_layer = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        self.camera = sim.player.astype(np.float32).copy()
        self.shake = 0.0
        self.particles: list[Particle] = []
        self.captions: list[Caption] = []
        self.effects: list[ScreenEffect] = []
        self.banner_text = ""
        self.banner_life = 0.0
        self.security_clear_life = 0.0
        self.room_label = ""
        self.room_label_life = 0.0
        self.high_contrast = False
        self.color_safe = False
        self.reduced_motion = False
        self.reduced_flashes = True
        self.sound_captions_enabled = True
        self.hud_scale = 1.0
        self.timer_warnings = True
        self.tutorial_hints = True
        self._role_cache: np.ndarray | None = None
        self._level_cache_identity: tuple[int, int, int] | None = None
        self._sprite_cache: dict[tuple[str, int, int, str], pygame.Surface] = {}
        self._atlas_scale_cache: dict[tuple[str, int], pygame.Surface] = {}
        self._character_scale_cache: dict[tuple[str, int, bool], pygame.Surface] = {}
        self._directional_gait_cache: dict[tuple[str, int, int], pygame.Surface] = {}
        self._diagonal_locomotion_cache: dict[tuple[str, int, int, bool], pygame.Surface] = {}
        self._last_room_role = ""
        self.font_small = pygame.font.SysFont("consolas", 10, bold=True)
        self.font = pygame.font.SysFont("consolas", 13, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 26, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 46, bold=True)
        self._hud_fonts = {
            scale: (
                pygame.font.SysFont("consolas", int(round(10 * scale)), bold=True),
                pygame.font.SysFont("consolas", int(round(13 * scale)), bold=True),
            )
            for scale in (1.0, 1.25, 1.5)
        }
        self._font_design_sizes = {
            id(self.font_small): 10,
            id(self.font): 13,
            id(self.font_large): 26,
            id(self.font_title): 46,
        }
        for scale, (small, regular) in self._hud_fonts.items():
            self._font_design_sizes[id(small)] = int(round(10 * scale))
            self._font_design_sizes[id(regular)] = int(round(13 * scale))
        self._native_text_commands: list[NativeTextCommand] = []
        self._native_font_cache: dict[int, pygame.font.Font] = {}
        self._clock = pygame.time.Clock()
        self._time = 0.0
        self._environment_atlas = self._load_environment_atlas()
        self._character_atlas = self._load_character_atlas()
        self._diagonal_locomotion_atlas = self._load_diagonal_locomotion_atlas()
        self.apply_accessibility(accessibility or {})

    def _load_environment_atlas(self) -> pygame.Surface | None:
        with runtime_asset_path("assets/visual/ghostline-environment-atlas-v1.png") as candidate:
            try:
                if candidate is not None:
                    source = pygame.image.load(str(candidate)).convert_alpha()
                    if source.get_size() != (1672, 941):
                        return None
                    return source
            except (OSError, pygame.error):
                pass
        return None

    def _load_character_atlas(self) -> pygame.Surface | None:
        with runtime_asset_path("assets/visual/ghostline-character-security-atlas-v1.png") as candidate:
            try:
                if candidate is not None:
                    source = pygame.image.load(str(candidate)).convert_alpha()
                    if source.get_size() != (1672, 941):
                        return None
                    return source
            except (OSError, pygame.error):
                pass
        return None

    def _load_diagonal_locomotion_atlas(self) -> pygame.Surface | None:
        with runtime_asset_path("assets/visual/ghostline-diagonal-locomotion-v2.png") as candidate:
            try:
                if candidate is not None:
                    source = pygame.image.load(str(candidate)).convert_alpha()
                    if source.get_size() != (1254, 1254):
                        return None
                    return source
            except (OSError, pygame.error):
                pass
        return None

    def _character_sprite(self, name: str, desired_height: int, *, flip: bool = False) -> pygame.Surface | None:
        if self._character_atlas is None or name not in CHARACTER_REGIONS:
            return None
        height = max(4, int(desired_height))
        key = (name, height, flip)
        cached = self._character_scale_cache.get(key)
        if cached is not None:
            return cached
        source = self._character_atlas.subsurface(pygame.Rect(CHARACTER_REGIONS[name].rect))
        width = max(4, int(round(source.get_width() * height / source.get_height())))
        sprite = pygame.transform.scale(source, (width, height))
        if flip:
            sprite = pygame.transform.flip(sprite, True, False)
        self._character_scale_cache[key] = sprite
        return sprite

    def _diagonal_locomotion_sprite(
        self,
        name: str,
        desired_height: int,
        *,
        flip: bool = False,
    ) -> pygame.Surface | None:
        if self._diagonal_locomotion_atlas is None or name not in DIAGONAL_LOCOMOTION_REGIONS:
            return None
        frame = 0 if self.reduced_motion else int(self._time * 10.0) % 4
        key = (name, desired_height, frame, flip)
        cached = self._diagonal_locomotion_cache.get(key)
        if cached is not None:
            return cached
        source = self._diagonal_locomotion_atlas.subsurface(
            pygame.Rect(DIAGONAL_LOCOMOTION_REGIONS[name][frame].rect)
        )
        width = max(1, round(source.get_width() * desired_height / source.get_height()))
        sprite = pygame.transform.scale(source, (width, desired_height))
        if flip:
            sprite = pygame.transform.flip(sprite, True, False)
        self._diagonal_locomotion_cache[key] = sprite
        return sprite

    def _atlas_sprite(self, kind: str, desired_width: int) -> pygame.Surface | None:
        if self._environment_atlas is None or kind not in ATLAS_REGIONS:
            return None
        region = ATLAS_REGIONS[kind]
        width = max(4, min(int(desired_width), region.maximum_world_width))
        key = (kind, width)
        cached = self._atlas_scale_cache.get(key)
        if cached is not None:
            return cached
        source = self._environment_atlas.subsurface(pygame.Rect(region.rect))
        height = max(4, int(round(source.get_height() * width / source.get_width())))
        # pygame.transform.scale is intentionally nearest-neighbour for pixel art.
        sprite = pygame.transform.scale(source, (width, height))
        self._atlas_scale_cache[key] = sprite
        return sprite

    def _blit_accessible_sprite(self, sprite: pygame.Surface, position: tuple[int, int], *, alpha: int = 255) -> None:
        image = sprite
        if alpha != 255:
            image = sprite.copy()
            image.set_alpha(alpha)
        if self.high_contrast:
            silhouette = pygame.mask.from_surface(image, threshold=8).to_surface(
                setcolor=(238, 250, 250, 255),
                unsetcolor=(0, 0, 0, 0),
            )
            for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
                self.logical.blit(silhouette, (position[0] + dx, position[1] + dy))
        self.logical.blit(image, position)

    def apply_accessibility(self, settings: dict[str, Any]) -> None:
        self.high_contrast = bool(settings.get("high_contrast", False))
        self.color_safe = bool(settings.get("color_safe", False))
        self.reduced_motion = bool(settings.get("reduced_motion", False))
        self.reduced_flashes = bool(settings.get("reduced_flashes", True))
        self.sound_captions_enabled = bool(settings.get("sound_captions", True))
        self.hud_scale = min((1.0, 1.25, 1.5), key=lambda value: abs(value - float(settings.get("hud_scale", 1.0))))
        self.timer_warnings = bool(settings.get("timer_warnings", True))
        self.tutorial_hints = bool(settings.get("tutorial_hints", True))

    def reset_for_sim(self, sim: GhostlineSimulation) -> None:
        """Attach a new deterministic simulation and clear presentation-only state."""

        self.sim = sim
        self.camera = sim.player.astype(np.float32).copy()
        self.particles.clear()
        self.captions.clear()
        self.effects.clear()
        self.banner_life = 0.0
        self.security_clear_life = 0.0
        self.room_label_life = 0.0
        self._level_cache_identity = None
        self._ensure_level_cache()
        tile = (int(sim.player[0] // TILE_SIZE), int(sim.player[1] // TILE_SIZE))
        self._last_room_role = self._room_role_at(*tile)

    def close(self) -> None:
        pygame.display.quit()

    def ingest_events(self, events: Iterable[SimEvent]) -> None:
        captions = {
            "dash": ("[FOOTSTEPS] Dash noise", CYAN),
            "dash_noise": ("[NOISE] Dash lure radius exposed", CYAN),
            "hack_tick": ("[TERMINAL] Link handshake", CYAN),
            "hack_complete": ("[TERMINAL] Data secured", AMBER),
            "quota_met": ("[OPERATOR] Quota met -- extract", GREEN),
            "detected": ("[ALARM] Position compromised", RED),
            "pulse": ("[PULSE] Electronics disrupted", CYAN),
            "damage": ("[IMPACT] Integrity damaged", RED),
            "lockdown": ("[ALARM] Lockdown protocol", RED),
            "extracted": ("[RELAY] Ghostline established", GREEN),
            "failure": ("[SYSTEM] Contract failed", RED),
            "drone_deployed": ("[ROTOR] Response drone inbound", VIOLET),
            "drone_warning": ("[NETWORK] Response drone threshold near", VIOLET),
            "guard_clear": ("[SECURITY] Search cleared", GREEN),
        }
        for event in events:
            position = np.asarray(event.position, dtype=np.float32)
            if event.kind in ("damage", "lockdown", "failure"):
                self.shake = max(self.shake, 4.0 if event.kind == "damage" else 2.0)
            if event.kind in captions and self.sound_captions_enabled:
                text, caption_color = captions[event.kind]
                if not self.captions or self.captions[-1].text != text:
                    self.captions.append(Caption(text, 2.1, caption_color))
                    self.captions = self.captions[-3:]
            if event.kind in ("dash_noise", "pulse", "damage", "quota_met", "lockdown", "extracted"):
                duration = 0.78 if event.kind == "dash_noise" else (0.65 if event.kind in ("pulse", "damage") else 1.25)
                if event.kind == "dash_noise":
                    self.effects = [effect for effect in self.effects if effect.kind != "dash_noise"]
                self.effects.append(ScreenEffect(event.kind, position, duration, duration))
                self.effects = self.effects[-12:]
            if event.kind == "guard_clear":
                self.security_clear_life = 1.35
            banner = {
                "quota_met": "QUOTA SECURED // REACH EXTRACTION",
                "lockdown": "LOCKDOWN // KEEP MOVING",
                "extracted": "GHOSTLINE ESTABLISHED",
            }.get(event.kind)
            if banner:
                self.banner_text, self.banner_life = banner, 2.0
            color = {
                "dash": CYAN,
                "dash_noise": CYAN,
                "hack_tick": CYAN,
                "hack_complete": AMBER,
                "quota_met": GREEN,
                "detected": RED,
                "pulse": CYAN,
                "damage": RED,
                "lockdown": RED,
                "extracted": GREEN,
                "drone_deployed": VIOLET,
                "drone_warning": VIOLET,
                "guard_clear": GREEN,
            }.get(event.kind)
            if color is None:
                continue
            if self.reduced_motion:
                continue
            if event.kind in ("dash", "hack_tick"):
                count = 2
            elif event.kind == "dash_noise":
                count = 4
            else:
                count = 9 if self.reduced_flashes else 18
            rng = np.random.default_rng(self.sim.elapsed_ticks + len(self.particles) * 19)
            for _ in range(count):
                angle = rng.uniform(0.0, math.tau)
                speed = rng.uniform(16.0, 72.0)
                self.particles.append(
                    Particle(position.copy(), np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32) * speed, color, rng.uniform(0.25, 0.8), int(rng.integers(1, 3)))
                )
            self.particles = self.particles[-72:]

    def draw(self, *, return_array: bool = False, lab_stats: dict[str, Any] | None = None) -> np.ndarray | bool:
        dt = min(0.05, self._clock.tick(60) / 1000.0) if self.visible else 1.0 / 60.0
        self._time += dt
        self._update_camera(dt)
        self._update_particles(dt)
        self._update_presentation_state(dt)
        self._ensure_level_cache()
        self._update_security_memory()
        self._native_text_commands.clear()
        self.logical.fill(BG)
        self._draw_floor()
        self._draw_security_cones()
        self._draw_walls()
        self._draw_props()
        self._draw_objectives()
        self._draw_security()
        self._draw_player()
        self._draw_particles()
        self._draw_lighting()
        self._draw_screen_effects()
        self._draw_alert_indicators()
        self._draw_objective_indicator()
        self._draw_hud(lab_stats)
        self._draw_detection_status()
        self._draw_captions()
        self._apply_accessibility_filter()
        return self._present(return_array=return_array)

    def draw_screen(
        self,
        *,
        title: str,
        subtitle: str = "",
        items: list[str] | None = None,
        selected: int = 0,
        body: list[str] | None = None,
        footer: str = "",
        panel: list[str] | None = None,
        badge: str = "",
        compact: bool = False,
        compact_body: bool = False,
        return_array: bool = False,
    ) -> np.ndarray | bool:
        self._time += self._clock.tick(60) / 1000.0
        self._native_text_commands.clear()
        self.logical.fill(BG)
        self._draw_menu_backdrop(title)
        self._text(title, 38, 48, self.font_title, INK)
        if badge:
            badge_width = self.font_small.size(badge)[0] + 14
            pygame.draw.rect(self.logical, (20, 48, 53), (42, 25, badge_width, 16), border_radius=2)
            pygame.draw.rect(self.logical, CYAN, (42, 25, badge_width, 16), 1, border_radius=2)
            self._text(badge, 49, 29, self.font_small, CYAN)
        if subtitle:
            self._text(subtitle.upper(), 42, 101, self.font_small, CYAN)
        if body:
            y = 132
            body_font = self.font_small if compact_body else self.font
            line_spacing = 16 if compact_body else 20
            for line in body:
                wrapped = self._wrap_text(line, body_font, 305 if panel else 550) if line else [""]
                for wrapped_line in wrapped:
                    self._text(wrapped_line, 42, y, body_font, INK if wrapped_line else MUTED)
                    y += line_spacing
        if items:
            y = 124 if compact else 137
            spacing = 18 if compact else 29
            for index, item in enumerate(items):
                active = index == selected
                if active:
                    pygame.draw.rect(self.logical, (20, 51, 55), (40, y - 4, 285, spacing - 5), border_radius=2)
                    pygame.draw.rect(self.logical, CYAN, (40, y - 4, 3, spacing - 5))
                self._text(("> " if active else "  ") + item, 49, y, self.font, CYAN if active else INK)
                y += spacing
        if panel:
            panel_lines = [
                wrapped
                for line in panel
                for wrapped in self._wrap_text(line, self.font_small, 208)
            ]
            # The right dossier owns the safe area above the footer. Wrap its
            # copy rather than letting long accessibility notes clip through
            # the panel edge, and expand vertically only as far as that safe
            # area permits.
            panel_rect = pygame.Rect(372, 106, 238, min(218, 31 + len(panel_lines) * 14))
            pygame.draw.rect(self.logical, (5, 12, 17), panel_rect, border_radius=4)
            pygame.draw.rect(self.logical, (46, 91, 94), panel_rect, 1, border_radius=4)
            pygame.draw.rect(self.logical, CYAN, (panel_rect.x, panel_rect.y, 3, panel_rect.height))
            self._text("LIVE DOSSIER", panel_rect.x + 14, panel_rect.y + 10, self.font_small, CYAN)
            for index, line in enumerate(panel_lines):
                self._text(line, panel_rect.x + 14, panel_rect.y + 29 + index * 14, self.font_small, INK if line else MUTED)
        if footer:
            self._text(footer, 42, 332, self.font_small, MUTED)
        self._apply_accessibility_filter()
        return self._present(return_array=return_array)

    def _present(self, *, return_array: bool) -> np.ndarray | bool:
        if return_array:
            return np.transpose(pygame.surfarray.array3d(self.logical), (1, 0, 2)).copy()
        target_size = self.window.get_size()
        scaled_size = _presentation_scaled_size(target_size)
        scaled = pygame.transform.scale(self.logical, scaled_size)
        self._draw_native_text(scaled, scaled_size)
        self.window.fill((1, 3, 5))
        destination = ((target_size[0] - scaled_size[0]) // 2, (target_size[1] - scaled_size[1]) // 2)
        self.window.blit(scaled, destination)
        pygame.display.flip()
        return True

    def _update_camera(self, dt: float) -> None:
        look = np.zeros(2, dtype=np.float32) if self.reduced_motion else self.sim.heading * 46.0 + self.sim.velocity * 0.14
        target = self.sim.player + look
        self.camera += (target - self.camera) * min(1.0, dt * 5.5)
        half_w, half_h = LOGICAL_SIZE[0] / 2, LOGICAL_SIZE[1] / 2
        self.camera[0] = np.clip(self.camera[0], half_w, max(half_w, self.sim.level.world_width - half_w))
        self.camera[1] = np.clip(self.camera[1], half_h, max(half_h, self.sim.level.world_height - half_h))
        self.shake = max(0.0, self.shake - dt * 12.0)

    def _world(self, position: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        shake = np.zeros(2)
        if self.screen_shake_enabled and not self.reduced_motion and self.shake > 0.0:
            shake = np.asarray((math.sin(self._time * 71), math.cos(self._time * 59))) * self.shake
        point = np.asarray(position) - self.camera + np.asarray((LOGICAL_SIZE[0] / 2, LOGICAL_SIZE[1] / 2)) + shake
        return int(round(point[0])), int(round(point[1]))

    def _visible_tile_bounds(self) -> tuple[int, int, int, int]:
        left = max(0, int((self.camera[0] - LOGICAL_SIZE[0] / 2) // TILE_SIZE) - 1)
        right = min(self.sim.level.grid.shape[1], int((self.camera[0] + LOGICAL_SIZE[0] / 2) // TILE_SIZE) + 2)
        top = max(0, int((self.camera[1] - LOGICAL_SIZE[1] / 2) // TILE_SIZE) - 1)
        bottom = min(self.sim.level.grid.shape[0], int((self.camera[1] + LOGICAL_SIZE[1] / 2) // TILE_SIZE) + 2)
        return left, right, top, bottom

    def _ensure_level_cache(self) -> None:
        identity = (self.sim.seed, self.sim.tier, id(self.sim.level))
        if self._level_cache_identity == identity and self._role_cache is not None:
            return
        cache = np.full(self.sim.level.grid.shape, "corridor", dtype=object)
        for room in self.sim.level.rooms:
            x1 = max(0, room.x)
            y1 = max(0, room.y)
            x2 = min(cache.shape[1], room.x + room.width + 1)
            y2 = min(cache.shape[0], room.y + room.height + 1)
            cache[y1:y2, x1:x2] = room.role
        self._role_cache = cache
        self._level_cache_identity = identity

    def _room_role_at(self, tile_x: int, tile_y: int) -> str:
        self._ensure_level_cache()
        if self._role_cache is not None and 0 <= tile_y < self._role_cache.shape[0] and 0 <= tile_x < self._role_cache.shape[1]:
            return str(self._role_cache[tile_y, tile_x])
        return "corridor"

    @staticmethod
    def _tile_hash(x: int, y: int, seed: int) -> int:
        return (x * 73_856_093 ^ y * 19_349_663 ^ seed * 83_492_791) & 0xFFFFFFFF

    def _draw_floor(self) -> None:
        left, right, top, bottom = self._visible_tile_bounds()
        for y in range(top, bottom):
            for x in range(left, right):
                if self.sim.level.grid[y, x] == Tile.WALL:
                    continue
                role = self._room_role_at(x, y)
                base = ROLE_COLORS.get(role, ROLE_COLORS["corridor"])
                shade = 4 if (x + y) % 2 else 0
                color = tuple(max(0, value - shade) for value in base)
                sx, sy = self._world(((x + 0.5) * TILE_SIZE, (y + 0.5) * TILE_SIZE))
                rect = pygame.Rect(sx - 16, sy - 16, 32, 32)
                pygame.draw.rect(self.logical, color, rect)
                pygame.draw.line(self.logical, tuple(max(0, c - 8) for c in color), rect.topleft, rect.topright)
                self._draw_floor_material(rect, role, self._tile_hash(x, y, self.sim.seed))
                if self.sim.level.grid[y, x] == Tile.DOOR:
                    pygame.draw.rect(self.logical, (8, 13, 17), rect)
                    pygame.draw.rect(self.logical, (52, 104, 105), rect, 2)
                    pygame.draw.line(self.logical, CYAN, (rect.x + 4, rect.centery), (rect.right - 4, rect.centery), 1)
                    pygame.draw.rect(self.logical, AMBER, (rect.x + 3, rect.y + 3, 3, 3))

    def _draw_floor_material(self, rect: pygame.Rect, role: str, value: int) -> None:
        """Deterministic room-specific material language, independent of simulation."""

        dark = (12, 19, 24)
        if role == "office":
            pygame.draw.line(self.logical, (44, 59, 68), (rect.x, rect.centery), (rect.right, rect.centery))
            if value % 7 == 0:
                pygame.draw.rect(self.logical, (55, 71, 78), (rect.x + 4, rect.y + 4, 5, 2))
        elif role == "lounge":
            inset = rect.inflate(-5, -5)
            pygame.draw.rect(self.logical, (61, 43, 64), inset, 1, border_radius=2)
            if value % 3 == 0:
                pygame.draw.line(self.logical, (78, 53, 77), inset.topleft, inset.bottomright)
        elif role == "lab":
            pygame.draw.rect(self.logical, (54, 73, 75), rect.inflate(-4, -4), 1)
            pygame.draw.circle(self.logical, (76, 103, 103), (rect.x + 6, rect.y + 6), 1)
        elif role == "server":
            for offset in (7, 15, 23):
                pygame.draw.line(self.logical, (22, 30, 44), (rect.x + offset, rect.y + 3), (rect.x + offset, rect.bottom - 3))
            if value % 5 == 0:
                pygame.draw.rect(self.logical, VIOLET, (rect.x + 4, rect.bottom - 5, 5, 1))
        elif role == "security":
            for offset in range(-16, 48, 10):
                pygame.draw.line(self.logical, (78, 48, 43), (rect.x + offset, rect.bottom), (rect.x + offset + 10, rect.y), 2)
        elif role == "vault":
            pygame.draw.rect(self.logical, (75, 66, 39), rect.inflate(-6, -6), 1)
            pygame.draw.circle(self.logical, (113, 91, 43), rect.center, 2, 1)
        elif role == "utility":
            pygame.draw.line(self.logical, (60, 65, 62), (rect.x + 4, rect.y + 4), (rect.right - 4, rect.bottom - 4))
            pygame.draw.circle(self.logical, dark, (rect.right - 6, rect.y + 6), 2)
        elif role == "extraction":
            pygame.draw.circle(self.logical, (39, 77, 65), rect.center, 10, 1)
        else:
            pygame.draw.line(self.logical, (49, 57, 64), (rect.centerx, rect.y), (rect.centerx, rect.bottom))
            if value % 4 == 0:
                pygame.draw.polygon(self.logical, (57, 69, 74), ((rect.x + 4, rect.centery), (rect.x + 10, rect.y + 10), (rect.x + 10, rect.bottom - 10)))
        if value % 17 == 0:
            pygame.draw.rect(self.logical, (88, 95, 89), (rect.x + 5, rect.y + 8, 2, 1))

    def _draw_walls(self) -> None:
        left, right, top, bottom = self._visible_tile_bounds()
        for y in range(top, bottom):
            for x in range(left, right):
                if self.sim.level.grid[y, x] != Tile.WALL:
                    continue
                sx, sy = self._world(((x + 0.5) * TILE_SIZE, (y + 0.5) * TILE_SIZE))
                rect = pygame.Rect(sx - 16, sy - 16, 32, 32)
                pygame.draw.rect(self.logical, (12, 19, 25), rect)
                pygame.draw.rect(self.logical, (25, 39, 47), (rect.x, rect.y, 32, 9))
                pygame.draw.line(self.logical, (56, 92, 101), rect.topleft, rect.topright)
                pygame.draw.line(self.logical, (5, 9, 13), rect.bottomleft, rect.bottomright, 2)
                role = self._room_role_at(x, y)
                detail = self._tile_hash(x, y, self.sim.seed)
                accent = {
                    "lab": CYAN,
                    "server": VIOLET,
                    "security": RED,
                    "vault": AMBER,
                    "extraction": GREEN,
                }.get(role, (51, 77, 83))
                pygame.draw.rect(self.logical, accent, (rect.x + 4, rect.y + 9, 24, 1))
                if detail % 5 == 0:
                    pygame.draw.rect(self.logical, (7, 12, 16), (rect.x + 7, rect.y + 14, 18, 9), border_radius=1)
                    for offset in (10, 15, 20):
                        pygame.draw.line(self.logical, (48, 67, 72), (rect.x + offset, rect.y + 16), (rect.x + offset, rect.y + 21))
                elif detail % 7 == 0:
                    pygame.draw.rect(self.logical, (42, 53, 57), (rect.x + 6, rect.y + 14, 20, 7), 1)
                    pygame.draw.rect(self.logical, accent, (rect.x + 9, rect.y + 16, 5, 2))

    def _draw_props(self) -> None:
        for prop in sorted(self.sim.level.props, key=lambda item: (item.tile_y + item.height, item.tile_x)):
            center = ((prop.tile_x + prop.width / 2) * TILE_SIZE, (prop.tile_y + prop.height / 2) * TILE_SIZE)
            sx, sy = self._world(center)
            width, height = prop.width * TILE_SIZE - 6, prop.height * TILE_SIZE - 6
            if not (-80 < sx < 720 and -80 < sy < 440):
                continue
            self._draw_prop(prop, pygame.Rect(sx - width // 2, sy - height // 2, width, height))

    def _draw_prop(self, prop: Prop, rect: pygame.Rect) -> None:
        atlas_sprite = self._atlas_sprite(prop.kind, rect.width)
        if atlas_sprite is not None:
            destination = (
                rect.centerx - atlas_sprite.get_width() // 2,
                rect.bottom - atlas_sprite.get_height() + 3,
            )
            self._blit_accessible_sprite(atlas_sprite, destination)
            return
        pygame.draw.rect(self.logical, (6, 10, 14), rect.move(3, 5), border_radius=2)
        kind = prop.kind
        if kind in ("desk", "meeting_table", "coffee_table", "lab_bench"):
            color = (83, 66, 61) if kind != "lab_bench" else (57, 84, 86)
            pygame.draw.rect(self.logical, color, rect, border_radius=2)
            pygame.draw.line(self.logical, tuple(min(255, c + 30) for c in color), rect.topleft, rect.topright)
            pygame.draw.rect(self.logical, (19, 28, 34), (rect.centerx - 5, rect.y + 4, 10, 6))
            pygame.draw.rect(self.logical, CYAN, (rect.centerx - 3, rect.y + 5, 6, 2))
            if rect.width > 32:
                pygame.draw.rect(self.logical, (130, 115, 92), (rect.x + 7, rect.y + 5, 7, 4))
                pygame.draw.line(self.logical, (178, 163, 132), (rect.x + 8, rect.y + 6), (rect.x + 12, rect.y + 6))
        elif kind == "chair":
            pygame.draw.rect(self.logical, (72, 62, 72), rect.inflate(-8, -5), border_radius=3)
            pygame.draw.line(self.logical, (113, 91, 109), (rect.x + 8, rect.y + 4), (rect.right - 8, rect.y + 4), 2)
        elif kind == "sofa":
            pygame.draw.rect(self.logical, (89, 54, 76), rect, border_radius=4)
            pygame.draw.rect(self.logical, (121, 69, 96), (rect.x + 3, rect.y + 3, rect.width - 6, 7), border_radius=2)
            pygame.draw.line(self.logical, (58, 38, 55), (rect.centerx, rect.y + 9), (rect.centerx, rect.bottom - 3))
        elif kind == "tv":
            pygame.draw.rect(self.logical, (8, 12, 18), rect, border_radius=2)
            pygame.draw.rect(self.logical, (22, 71, 78), rect.inflate(-6, -8))
            pygame.draw.line(self.logical, CYAN, (rect.x + 5, rect.y + 6), (rect.right - 8, rect.bottom - 7))
            scan_y = rect.y + 4 + int((self._time * 11) % max(2, rect.height - 8))
            pygame.draw.line(self.logical, (61, 145, 147), (rect.x + 4, scan_y), (rect.right - 4, scan_y))
        elif kind in ("server", "locker"):
            color = (31, 42, 54) if kind == "server" else (51, 55, 62)
            pygame.draw.rect(self.logical, color, rect, border_radius=2)
            for y in range(rect.y + 5, rect.bottom - 3, 7):
                pygame.draw.line(self.logical, (77, 99, 112), (rect.x + 4, y), (rect.right - 4, y))
                if kind == "server":
                    lit = int(self._time * 5 + y + prop.tile_x) % 3
                    pygame.draw.rect(self.logical, (CYAN if y % 2 else VIOLET) if lit else (34, 54, 60), (rect.right - 7, y - 1, 2, 2))
        elif kind in ("console", "monitor"):
            pygame.draw.rect(self.logical, (18, 29, 36), rect, border_radius=2)
            inner = rect.inflate(-7, -7)
            pygame.draw.rect(self.logical, (24, 82, 85), inner)
            pygame.draw.line(self.logical, CYAN, inner.topleft, inner.bottomright)
            signal = int((math.sin(self._time * 3.0 + prop.tile_y) + 1.0) * max(1, inner.width - 3) / 2)
            pygame.draw.rect(self.logical, INK, (inner.x + min(inner.width - 2, signal), inner.bottom - 3, 2, 2))
        elif kind == "plant":
            pygame.draw.rect(self.logical, (82, 57, 43), (rect.centerx - 5, rect.centery, 10, 9))
            for angle in (-2.4, -1.8, -1.2, -0.6):
                tip = (rect.centerx + math.cos(angle) * 10, rect.centery + math.sin(angle) * 12)
                pygame.draw.line(self.logical, (66, 143, 93), (rect.centerx, rect.centery + 2), tip, 3)
        elif kind in ("crate", "generator", "vault_case"):
            color = (91, 70, 44) if kind == "crate" else ((57, 70, 71) if kind == "generator" else (102, 82, 42))
            pygame.draw.rect(self.logical, color, rect, border_radius=2)
            pygame.draw.rect(self.logical, tuple(min(255, c + 28) for c in color), rect, 2)
            pygame.draw.line(self.logical, (20, 25, 28), rect.topleft, rect.bottomright, 2)
            if kind == "generator":
                pygame.draw.circle(self.logical, AMBER, rect.center, min(rect.width, rect.height) // 5, 2)
                pygame.draw.arc(self.logical, CYAN, rect.inflate(-8, -8), self._time * 2.0, self._time * 2.0 + 1.8, 2)
        else:
            # Unknown authored props still receive a deliberate in-world treatment.
            pygame.draw.rect(self.logical, (51, 63, 67), rect, border_radius=2)
            pygame.draw.rect(self.logical, (79, 98, 101), rect, 1, border_radius=2)
            pygame.draw.line(self.logical, (26, 35, 40), rect.topleft, rect.bottomright)

    def _draw_objectives(self) -> None:
        for terminal in self.sim.level.terminals:
            sx, sy = self._world(terminal.position)
            color = (59, 70, 74) if terminal.completed else AMBER
            pulse = 2 + int(2 * (0.5 + 0.5 * math.sin(self._time * 4 + terminal.terminal_id)))
            terminal_sprite = self._atlas_sprite("terminal", 28)
            if terminal_sprite is not None:
                self._blit_accessible_sprite(
                    terminal_sprite,
                    (sx - terminal_sprite.get_width() // 2, sy - terminal_sprite.get_height() + 10),
                    alpha=105 if terminal.completed else 255,
                )
            else:
                pygame.draw.rect(self.logical, (9, 14, 19), (sx - 10, sy - 10, 20, 20), border_radius=3)
                pygame.draw.rect(self.logical, color, (sx - 8, sy - 8, 16, 16), 2, border_radius=2)
                pygame.draw.rect(self.logical, color, (sx - 3, sy - 3, 6, 6))
            if not terminal.completed:
                in_range = norm(terminal.position - self.sim.player) <= HACK_RADIUS
                zone_color = CYAN if in_range else color
                # The outer ring is literal interaction geometry.  Keeping it
                # tied to HACK_RADIUS prevents a player from appearing outside
                # the zone while a valid moving link continues.
                pygame.draw.circle(
                    self.logical,
                    zone_color,
                    (sx, sy),
                    int(round(HACK_RADIUS)),
                    1,
                )
                ring_pulse = 15 if self.reduced_motion else 13 + pulse
                pygame.draw.circle(self.logical, (*color, 80), (sx, sy), ring_pulse, 1)
                self._text(str(terminal.value), sx - 3, sy - 20, self.font_small, AMBER)
                if terminal.progress > 0.0:
                    progress = min(1.0, terminal.progress / max(0.001, terminal.hack_seconds))
                    pygame.draw.arc(self.logical, CYAN, (sx - 24, sy - 24, 48, 48), -math.pi / 2, -math.pi / 2 + progress * math.tau, 3)
                if in_range:
                    link_label = f"LINK {terminal.hack_seconds:.1f}s"
                    self._text(link_label, sx - self.font_small.size(link_label)[0] // 2, sy + 25, self.font_small, CYAN)
        sx, sy = self._world(self.sim.level.extraction)
        color = GREEN if self.sim.quota_met else (75, 88, 88)
        pygame.draw.circle(self.logical, color, (sx, sy), 19, 2)
        extraction_pulse = 13 if self.reduced_motion else 12 + int(3 * math.sin(self._time * 3))
        pygame.draw.circle(self.logical, color, (sx, sy), extraction_pulse, 1)
        for angle in np.linspace(0.0, math.tau, 4, endpoint=False):
            point = (sx + math.cos(angle + self._time) * 15, sy + math.sin(angle + self._time) * 15)
            pygame.draw.circle(self.logical, color, point, 2)

    def _draw_security_cones(self) -> None:
        layer = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        for camera in self.sim.level.cameras:
            if camera.disabled_for > 0.0 or not self._visible_from_player(camera.position):
                continue
            pressure = 1.0 if camera.detected else camera.awareness
            self._cone(
                layer,
                camera.position,
                camera.angle,
                CAMERA_VIEW_DISTANCE,
                CAMERA_VIEW_HALF_ANGLE,
                self._vision_color(pressure),
                14 + int(30 * pressure),
                pattern="camera",
                pressure=pressure,
            )
        for guard in self.sim.level.guards:
            if not self._visible_from_player(guard.position):
                continue
            pressure = 1.0 if guard.mode == GuardMode.CHASE else guard.awareness
            self._cone(
                layer,
                guard.position,
                guard.facing,
                GUARD_VIEW_BASE_DISTANCE + GUARD_VIEW_ALERT_DISTANCE * self.sim.alert_tier,
                GUARD_VIEW_HALF_ANGLE,
                self._vision_color(pressure),
                10 + int(28 * pressure),
                pattern="guard",
                pressure=pressure,
            )
        self.logical.blit(layer, (0, 0))

    @staticmethod
    def _vision_color(pressure: float) -> tuple[int, int, int]:
        """Return a three-state danger palette; shape cues carry the same state."""

        pressure = float(np.clip(pressure, 0.0, 1.0))
        if pressure < 0.35:
            return AMBER
        blend = min(1.0, (pressure - 0.35) / 0.65)
        return tuple(int(round(a + (b - a) * blend)) for a, b in zip(AMBER, RED))

    def _cone_raycast_distance(self, origin: np.ndarray, direction: np.ndarray, maximum: float) -> float:
        """Stop a presentation ray at the first simulation occluder.

        Tile-boundary DDA prevents the fan polygon from bleeding across a
        wall/prop edge without sampling every few pixels.  Detection continues
        to use ``simulation.visible``; this is its fast presentation mask.
        """

        grid = self.sim.level.grid
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = float(direction[0]), float(direction[1])
        tile_x, tile_y = int(ox // TILE_SIZE), int(oy // TILE_SIZE)
        step_x = 1 if dx >= 0.0 else -1
        step_y = 1 if dy >= 0.0 else -1
        if abs(dx) < 1e-8:
            next_x = math.inf
            delta_x = math.inf
        else:
            boundary_x = (tile_x + (1 if step_x > 0 else 0)) * TILE_SIZE
            next_x = (boundary_x - ox) / dx
            delta_x = TILE_SIZE / abs(dx)
        if abs(dy) < 1e-8:
            next_y = math.inf
            delta_y = math.inf
        else:
            boundary_y = (tile_y + (1 if step_y > 0 else 0)) * TILE_SIZE
            next_y = (boundary_y - oy) / dy
            delta_y = TILE_SIZE / abs(dy)

        while min(next_x, next_y) <= maximum:
            if next_x < next_y:
                entry_distance = next_x
                tile_x += step_x
                next_x += delta_x
            elif next_y < next_x:
                entry_distance = next_y
                tile_y += step_y
                next_y += delta_y
            else:
                entry_distance = next_x
                tile_x += step_x
                tile_y += step_y
                next_x += delta_x
                next_y += delta_y
            if not (0 <= tile_y < grid.shape[0] and 0 <= tile_x < grid.shape[1]):
                return max(0.0, min(maximum, entry_distance) - 0.75)
            if (tile_x, tile_y) in self.sim._blocked_tiles or grid[tile_y, tile_x] == Tile.WALL:
                return max(0.0, min(maximum, entry_distance) - 0.75)
        return maximum

    def _cone_points(
        self,
        origin: np.ndarray,
        angle: float,
        distance: float,
        half_angle: float,
        *,
        ray_count: int = 65,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        angles = np.linspace(angle - half_angle, angle + half_angle, max(3, ray_count))
        distances: list[float] = []
        points = [self._world(origin)]
        for ray_angle in angles:
            direction = angle_vector(float(ray_angle))
            ray_distance = self._cone_raycast_distance(origin, direction, distance)
            distances.append(ray_distance)
            points.append(self._world(origin + direction * ray_distance))
        return points, distances

    def _cone(
        self,
        layer: pygame.Surface,
        origin: np.ndarray,
        angle: float,
        distance: float,
        half_angle: float,
        color: tuple[int, int, int],
        alpha: int,
        *,
        pattern: str = "guard",
        pressure: float = 0.0,
    ) -> None:
        center = self._world(origin)
        points, ray_distances = self._cone_points(origin, angle, distance, half_angle)
        pygame.draw.polygon(layer, (*color, alpha), points)
        edge = (*color, min(150, 66 + int(76 * pressure)))
        width = 2 if pressure >= 0.55 else 1
        pygame.draw.line(layer, edge, center, points[1], width)
        pygame.draw.line(layer, edge, center, points[-1], width)

        # Electronic scans use a dashed centre beam; human sight uses paired
        # boundary notches.  The pattern remains legible when danger colors are
        # remapped and avoids covering the room with a solid translucent wedge.
        middle = len(ray_distances) // 2
        if pattern == "camera":
            center_distance = ray_distances[middle]
            direction = angle_vector(angle)
            dash_alpha = min(190, 92 + int(90 * pressure))
            for start in np.arange(15.0, center_distance, 18.0):
                end = min(center_distance, float(start) + 8.0)
                pygame.draw.line(
                    layer,
                    (*color, dash_alpha),
                    self._world(origin + direction * float(start)),
                    self._world(origin + direction * end),
                    1 if pressure < 0.65 else 2,
                )
        else:
            for point_index in (1, len(points) - 1):
                endpoint = np.asarray(points[point_index], dtype=np.float32)
                vector = endpoint - np.asarray(center, dtype=np.float32)
                magnitude = max(1.0, float(np.linalg.norm(vector)))
                along = vector / magnitude
                lateral = np.asarray((-along[1], along[0]))
                for fraction in (0.42, 0.7):
                    notch = np.asarray(center, dtype=np.float32) + vector * fraction
                    pygame.draw.line(
                        layer,
                        (*color, min(145, 76 + int(64 * pressure))),
                        (notch - lateral * 2.5).astype(int),
                        (notch + lateral * 2.5).astype(int),
                        1,
                    )

    @staticmethod
    def _draw_threat_glyph(
        surface: pygame.Surface,
        kind: str,
        center: tuple[int, int],
        color: tuple[int, int, int],
        *,
        filled: bool = False,
    ) -> None:
        """Draw semantic shapes that survive monochrome/color-safe play."""

        x, y = center
        width = 0 if filled else 1
        if kind == "camera":
            pygame.draw.rect(surface, color, (x - 4, y - 4, 8, 8), width)
            pygame.draw.circle(surface, (6, 11, 15) if filled else color, (x, y), 1)
        elif kind == "sound":
            pygame.draw.arc(surface, color, (x - 5, y - 5, 7, 10), -math.pi / 2, math.pi / 2, 1)
            pygame.draw.arc(surface, color, (x - 5, y - 7, 11, 14), -math.pi / 2, math.pi / 2, 1)
        else:
            pygame.draw.polygon(surface, color, ((x, y - 5), (x + 5, y + 4), (x - 5, y + 4)), width)

    def _draw_awareness_badge(
        self,
        x: int,
        y: int,
        *,
        kind: str,
        pressure: float,
        confirmed: bool = False,
        suspicious: bool = False,
    ) -> None:
        """Render acquire progress and AI state above a visible security source."""

        pressure = float(np.clip(pressure, 0.0, 1.0))
        if pressure <= 0.01 and not confirmed and not suspicious:
            return
        color = self._vision_color(1.0 if confirmed else pressure)
        if confirmed:
            pygame.draw.rect(self.logical, (4, 8, 12), (x - 13, y - 7, 26, 14), border_radius=2)
            pygame.draw.rect(self.logical, color, (x - 13, y - 7, 26, 14), 1, border_radius=2)
            self._draw_threat_glyph(self.logical, kind, (x - 6, y), color, filled=True)
            self._text("!", x + 3, y - 5, self.font_small, color)
            return
        if pressure <= 0.01:
            self._draw_threat_glyph(self.logical, kind, (x - 4, y), AMBER)
            self._text("?", x + 3, y - 5, self.font_small, AMBER)
            return

        pygame.draw.rect(self.logical, (4, 8, 12), (x - 19, y - 5, 38, 10), border_radius=2)
        self._draw_threat_glyph(self.logical, kind, (x - 13, y), color)
        filled_segments = max(1, int(math.ceil(pressure * 4.0)))
        for index in range(4):
            pygame.draw.rect(
                self.logical,
                color if index < filled_segments else (42, 53, 56),
                (x - 5 + index * 6, y - 2, 4, 4),
            )

    def _draw_security(self) -> None:
        for camera in self.sim.level.cameras:
            if not self._visible_from_player(camera.position):
                continue
            sx, sy = self._world(camera.position)
            color = CYAN if camera.disabled_for > 0.0 else (RED if camera.detected else AMBER)
            camera_sprite = self._camera_atlas_sprite(
                disabled_for=camera.disabled_for,
                detected=camera.detected,
                awareness=camera.awareness,
            )
            character_camera = camera_sprite is not None
            if camera_sprite is None:
                camera_sprite = self._atlas_sprite("camera", 24)
            if camera_sprite is not None:
                rotated = camera_sprite if character_camera else pygame.transform.rotate(camera_sprite, -math.degrees(camera.angle) - 22.5)
                self._blit_accessible_sprite(
                    rotated,
                    (sx - rotated.get_width() // 2, sy - rotated.get_height() // 2),
                    alpha=145 if camera.disabled_for > 0.0 else 255,
                )
            else:
                pygame.draw.circle(self.logical, (10, 16, 21), (sx, sy), 8)
            pygame.draw.line(self.logical, color, (sx, sy), (sx + math.cos(camera.angle) * 10, sy + math.sin(camera.angle) * 10), 3)
            pygame.draw.circle(self.logical, color, (sx, sy), 5 if camera_sprite is None else 3, 2)
            self._draw_awareness_badge(
                sx,
                sy - 21,
                kind="camera",
                pressure=camera.awareness,
                confirmed=camera.detected,
            )
        for guard in self.sim.level.guards:
            if not self._visible_from_player(guard.position):
                continue
            self._draw_guard(
                guard.position,
                guard.facing,
                guard.mode,
                guard.velocity,
                guard.attack_windup,
                guard.grade,
            )
            sx, sy = self._world(guard.position)
            self._draw_awareness_badge(
                sx,
                sy - 27,
                kind="guard",
                pressure=guard.awareness,
                confirmed=guard.mode == GuardMode.CHASE,
                suspicious=guard.mode in (GuardMode.SUSPICIOUS, GuardMode.INVESTIGATE, GuardMode.SEARCH),
            )
            if guard.mode not in (GuardMode.PATROL, GuardMode.RETURN):
                cause = {"eye": "EYE", "sound": "SOUND", "radio": "RADIO"}.get(guard.stimulus, "CHECK")
                cause_color = RED if guard.stimulus == "eye" else (CYAN if guard.stimulus == "sound" else VIOLET)
                width = self.font_small.size(cause)[0] + 6
                cause_y = sy - 47 if sy - 47 >= 116 else sy + 26
                pygame.draw.rect(self.logical, (4, 8, 12), (sx - width // 2, cause_y, width, 10), border_radius=2)
                self._text(cause, sx - width // 2 + 3, cause_y + 1, self.font_small, cause_color)
        for drone in self.sim.drones:
            if not self._visible_from_player(drone.position):
                continue
            sx, sy = self._world(drone.position)
            color = CYAN if drone.disabled_for > 0.0 else VIOLET
            drone_sprite = self._drone_atlas_sprite(drone.facing, drone.disabled_for, drone.attack_windup)
            if drone_sprite is not None:
                self._blit_accessible_sprite(
                    drone_sprite,
                    (sx - drone_sprite.get_width() // 2, sy + 10 - drone_sprite.get_height()),
                    alpha=155 if drone.disabled_for > 0.0 else 255,
                )
                pygame.draw.circle(self.logical, color, (sx, sy), 3, 1)
            else:
                pygame.draw.circle(self.logical, (9, 13, 19), (sx, sy), 10)
                pygame.draw.polygon(self.logical, color, ((sx, sy - 8), (sx + 9, sy), (sx, sy + 8), (sx - 9, sy)), 2)
                pygame.draw.circle(self.logical, color, (sx, sy), 2)
            if drone.attack_windup > 0.0:
                progress = min(1.0, drone.attack_windup / DRONE_STRIKE_WINDUP_SECONDS)
                radius = max(13, int(28 - progress * 14))
                cue = (255, 255, 255) if progress > 0.76 and not self.reduced_flashes else VIOLET
                points = [
                    (
                        int(sx + math.cos(index * math.tau / 6.0) * radius),
                        int(sy + math.sin(index * math.tau / 6.0) * radius),
                    )
                    for index in range(6)
                ]
                pygame.draw.polygon(self.logical, cue, points, 2)
                pygame.draw.arc(
                    self.logical,
                    cue,
                    (sx - 19, sy - 19, 38, 38),
                    -math.pi / 2,
                    -math.pi / 2 + math.tau * progress,
                    3,
                )
                pygame.draw.line(self.logical, cue, (sx, sy), self._world(self.sim.player), 1)
                self._text("CHARGE", sx - 18, sy - 35, self.font_small, cue)

    def _update_security_memory(self) -> None:
        """Refresh the renderer-independent, player-earned intel cache."""

        self.sim._update_player_intel()

    @property
    def _security_memory(self):
        """Compatibility alias for tests and minimap drawing."""

        return self.sim.security_intel

    def _visible_from_player(
        self,
        position: np.ndarray,
        *,
        distance: float = PLAYER_PERCEPTION_DISTANCE,
    ) -> bool:
        del distance
        return self.sim.player_can_track_security(position)

    @staticmethod
    def _direction_index(facing: float) -> int:
        return int(round((facing % math.tau) / (math.tau / 8.0))) % 8

    def _runner_atlas_sprite(self, facing: float, moving: bool, state: str) -> pygame.Surface | None:
        direction = self._direction_index(facing)
        face_left = math.cos(facing) < -0.08
        if state == "dash":
            return self._character_sprite("runner_dash", 28, flip=face_left)
        if state == "hack":
            return self._character_sprite("runner_link", 34, flip=face_left)
        if state == "damage":
            return self._character_sprite("runner_damage", 29, flip=face_left)
        # The authored side-on run strip only reads correctly east/west.  Using
        # it for diagonal input made the runner visibly snap out of their
        # travel direction.  Directional locomotion keeps the facing pose and
        # animates its torso/individual legs on the same integer-pixel pivot.
        if moving and direction in (1, 3, 5, 7):
            diagonal = "runner_se" if direction in (1, 3) else "runner_ne"
            authored = self._diagonal_locomotion_sprite(diagonal, 32, flip=direction in (3, 5))
            if authored is not None:
                return authored
        if moving and direction in (1, 2, 3, 5, 6, 7):
            region, flip = RUNNER_DIRECTION_REGIONS[direction]
            return self._directional_gait_sprite(region, direction, desired_height=32, flip=flip)
        if moving and direction in (0, 4):
            frame = ("runner_run_a", "runner_run_b", "runner_run_c")[int(self._time * 9.0) % 3]
            return self._character_sprite(frame, 31, flip=face_left)
        region, flip = RUNNER_DIRECTION_REGIONS[direction]
        return self._character_sprite(region, 32, flip=flip)

    def _guard_atlas_sprite(
        self,
        facing: float,
        mode: GuardMode,
        moving: bool,
        attack_windup: float,
    ) -> pygame.Surface | None:
        direction = self._direction_index(facing)
        face_left = math.cos(facing) < -0.08
        if attack_windup > 0.0:
            return self._character_sprite("guard_strike", 32, flip=face_left)
        if mode == GuardMode.CHASE and direction in (0, 4):
            frame = "guard_chase" if int(self._time * 8.0) % 2 else "guard_run"
            return self._character_sprite(frame, 34, flip=face_left)
        if mode in (GuardMode.SUSPICIOUS, GuardMode.INVESTIGATE, GuardMode.SEARCH):
            return self._character_sprite("guard_suspicious", 34, flip=face_left)
        if moving and direction in (1, 3, 5, 7):
            diagonal = "guard_se" if direction in (1, 3) else "guard_ne"
            authored = self._diagonal_locomotion_sprite(diagonal, 34, flip=direction in (3, 5))
            if authored is not None:
                return authored
        if moving and direction in (1, 2, 3, 5, 6, 7):
            region, flip = GUARD_DIRECTION_REGIONS[direction]
            return self._directional_gait_sprite(region, direction, desired_height=34, flip=flip)
        if moving and direction in (0, 4):
            return self._character_sprite("guard_walk", 34, flip=face_left)
        region, flip = GUARD_DIRECTION_REGIONS[direction]
        return self._character_sprite(region, 34, flip=flip)

    def _directional_gait_sprite(
        self,
        region: str,
        direction: int,
        *,
        desired_height: int,
        flip: bool,
    ) -> pygame.Surface | None:
        """Animate an atlas pose without losing its eight-way silhouette.

        The lower sprite is split into two independently moving leg blocks.
        All transforms are integer nearest-neighbour operations, so the loop
        stays crisp on the 640x360 logical canvas and keeps a stable foot pivot.
        """

        phase = 0 if self.reduced_motion else int(self._time * 10.0) % 4
        key = (region, desired_height, phase | (int(flip) << 3))
        cached = self._directional_gait_cache.get(key)
        if cached is not None:
            return cached
        base = self._character_sprite(region, desired_height, flip=flip)
        if base is None:
            return None

        width, height = base.get_size()
        canvas = pygame.Surface((width + 4, height + 4), pygame.SRCALPHA)
        pivot_x = 2
        bob = (0, -1, 0, 0)[phase]
        # A one-pixel forward lean distinguishes the contact and recoil poses.
        forward_x = int(round(math.cos(direction * math.tau / 8.0)))
        lean = (0, forward_x, 0, -forward_x)[phase]
        body_y = height - max(8, height // 3)
        canvas.blit(base, (pivot_x + lean, 1 + bob), area=pygame.Rect(0, 0, width, body_y))

        leg_height = height - body_y
        split = max(1, width // 2)
        stride = (1, 0, -1, 0)[phase]
        left = pygame.Rect(0, body_y, split, leg_height)
        right = pygame.Rect(split, body_y, width - split, leg_height)
        canvas.blit(base, (pivot_x, 1 + body_y + stride), area=left)
        canvas.blit(base, (pivot_x + split, 1 + body_y - stride), area=right)

        # Preserve the bottom pivot even while one leg recoils upward.
        if stride:
            contact = right if stride > 0 else left
            destination_x = pivot_x + (split if stride > 0 else 0)
            canvas.blit(base, (destination_x, 1 + body_y), area=contact)
        self._directional_gait_cache[key] = canvas
        return canvas

    def _drone_atlas_sprite(self, facing: float, disabled_for: float, attack_windup: float) -> pygame.Surface | None:
        if disabled_for > 0.0:
            return self._character_sprite("drone_recoil", 27)
        if attack_windup > 0.0:
            return self._character_sprite("drone_charge", 31)
        direction = self._direction_index(facing)
        region, flip = DRONE_DIRECTION_REGIONS[direction]
        return self._character_sprite(region, 21, flip=flip)

    def _camera_atlas_sprite(self, *, disabled_for: float, detected: bool, awareness: float) -> pygame.Surface | None:
        if disabled_for > 0.0:
            name = "camera_disabled"
        elif detected:
            name = "camera_detected"
        elif awareness > 0.05:
            name = "camera_suspicious"
        else:
            name = "camera_active"
        return self._character_sprite(name, 24)

    def _draw_guard(
        self,
        position: np.ndarray,
        facing: float,
        mode: GuardMode,
        velocity: np.ndarray,
        attack_windup: float = 0.0,
        grade: GuardGrade = GuardGrade.STANDARD,
    ) -> None:
        sx, sy = self._world(position)
        calm = mode in (GuardMode.PATROL, GuardMode.RETURN)
        state = "chase" if mode == GuardMode.CHASE else ("normal" if calm else "alert")
        moving = norm(velocity) > 5.0
        sprite = self._guard_atlas_sprite(facing, mode, moving, attack_windup)
        if sprite is not None:
            self._blit_accessible_sprite(sprite, (sx - sprite.get_width() // 2, sy + 13 - sprite.get_height()))
        else:
            sprite = self._actor_sprite("guard", facing, moving, state)
            self.logical.blit(sprite, (sx - sprite.get_width() // 2, sy - 17))
        if not calm:
            symbol = "!" if mode == GuardMode.CHASE else "?"
            symbol_y = sy - 22 if sy - 22 >= 116 else sy + 48
            self._text(symbol, sx - 3, symbol_y, self.font, RED if symbol == "!" else AMBER)
        grade_color = (142, 166, 171) if grade == GuardGrade.STANDARD else (AMBER if grade == GuardGrade.INTERCEPTOR else RED)
        grade_label = ("I", "II", "III")[int(grade)]
        label_width = self.font_small.size(grade_label)[0]
        pygame.draw.rect(self.logical, (4, 8, 12), (sx - label_width // 2 - 3, sy + 12, label_width + 6, 10), border_radius=2)
        self._text(grade_label, sx - label_width // 2, sy + 12, self.font_small, grade_color)
        if attack_windup > 0.0:
            progress = min(1.0, attack_windup / GUARD_STRIKE_WINDUP_SECONDS)
            radius = max(12, int(25 - progress * 11))
            cue_color = (255, 255, 255) if progress > 0.72 and not self.reduced_flashes else RED
            pygame.draw.polygon(
                self.logical,
                cue_color,
                ((sx, sy - radius), (sx + radius, sy), (sx, sy + radius), (sx - radius, sy)),
                2,
            )
            pygame.draw.arc(self.logical, cue_color, (sx - 18, sy - 18, 36, 36), -math.pi / 2, -math.pi / 2 + math.tau * progress, 3)
            player_screen = self._world(self.sim.player)
            pygame.draw.line(self.logical, (*cue_color, 90), (sx, sy), player_screen, 1)
            strike_y = sy - 34 if sy - 34 >= 116 else sy + 39
            self._text("STRIKE", sx - 18, strike_y, self.font_small, cue_color)

    def _draw_player(self) -> None:
        sx, sy = self._world(self.sim.player)
        facing = math.atan2(float(self.sim.heading[1]), float(self.sim.heading[0]))
        if self.effects and any(effect.kind == "damage" for effect in self.effects):
            state = "damage"
        elif self.sim.active_hack_progress > 0.0:
            state = "hack"
        elif norm(self.sim.velocity) > 170.0:
            state = "dash"
        else:
            state = "normal"
        moving = norm(self.sim.velocity) > 5.0
        sprite = self._runner_atlas_sprite(facing, moving, state)
        atlas_sprite = sprite is not None
        if sprite is None:
            sprite = self._actor_sprite("runner", facing, moving, state)
        destination = (sx - sprite.get_width() // 2, sy + 13 - sprite.get_height()) if atlas_sprite else (sx - sprite.get_width() // 2, sy - 17)
        if state == "dash" and not self.reduced_motion:
            direction = self.sim.heading
            for distance, alpha in ((15, 90), (27, 45)):
                ghost = sprite.copy()
                ghost.set_alpha(alpha)
                self.logical.blit(
                    ghost,
                    (
                        int(round(destination[0] - float(direction[0]) * distance)),
                        int(round(destination[1] - float(direction[1]) * distance)),
                    ),
                )
        if atlas_sprite:
            self._blit_accessible_sprite(sprite, destination)
        else:
            self.logical.blit(sprite, destination)

    def _actor_sprite(self, kind: str, facing: float, moving: bool, state: str) -> pygame.Surface:
        """Build and cache an original eight-direction, four-frame pixel actor."""

        direction = int(round((facing % math.tau) / (math.tau / 8.0))) % 8
        frame = int(self._time * 8.0) % 4 if moving and not self.reduced_motion else 0
        key = (kind, direction, frame, state)
        cached = self._sprite_cache.get(key)
        if cached is not None:
            return cached

        surface = pygame.Surface((26, 30), pygame.SRCALPHA)
        angle = direction * math.tau / 8.0
        forward = np.asarray((math.cos(angle), math.sin(angle)))
        lateral = np.asarray((-forward[1], forward[0]))
        gait = (0, 2, 0, -2)[frame]
        shadow = pygame.Rect(4, 23, 18, 5)
        pygame.draw.ellipse(surface, (2, 5, 8, 150), shadow)

        if kind == "runner":
            body, dark, skin, accent = (29, 50, 56), (13, 25, 30), (197, 151, 119), CYAN
            if state == "dash":
                accent = (129, 255, 240)
            elif state == "hack":
                accent = AMBER
            elif state == "damage":
                accent = RED
        else:
            body, dark, skin = (47, 53, 62), (22, 27, 34), (190, 139, 107)
            accent = RED if state == "chase" else (AMBER if state == "alert" else (176, 190, 194))

        center = np.asarray((13.0, 15.0))
        feet_center = center + forward * 4
        foot_a = feet_center + lateral * (3 + gait * 0.35) - forward * gait
        foot_b = feet_center - lateral * (3 + gait * 0.35) + forward * gait
        pygame.draw.line(surface, dark, center + lateral * 2, foot_a + np.asarray((0, 7)), 3)
        pygame.draw.line(surface, dark, center - lateral * 2, foot_b + np.asarray((0, 7)), 3)
        pygame.draw.rect(surface, (5, 9, 13), (int(foot_a[0] - 2), int(foot_a[1] + 6), 5, 2))
        pygame.draw.rect(surface, (5, 9, 13), (int(foot_b[0] - 2), int(foot_b[1] + 6), 5, 2))

        shoulders = center - forward * 2
        body_points = [
            shoulders + lateral * 6,
            shoulders - lateral * 6,
            center + forward * 7 - lateral * 4,
            center + forward * 7 + lateral * 4,
        ]
        pygame.draw.polygon(surface, body, [(int(point[0]), int(point[1])) for point in body_points])
        pygame.draw.line(surface, accent, shoulders + lateral * 5, shoulders - lateral * 5, 2)
        arm_swing = gait * 0.45
        pygame.draw.line(surface, skin, shoulders + lateral * 6, center + lateral * 7 + forward * arm_swing, 2)
        pygame.draw.line(surface, skin, shoulders - lateral * 6, center - lateral * 7 - forward * arm_swing, 2)

        head = center - forward * 8
        pygame.draw.circle(surface, skin, (int(head[0]), int(head[1])), 5)
        pygame.draw.arc(surface, dark, (int(head[0] - 5), int(head[1] - 5), 10, 9), math.pi, math.tau, 2)
        visor_start = head + lateral * 3 + forward * 2
        visor_end = head - lateral * 3 + forward * 2
        pygame.draw.line(surface, accent, visor_start, visor_end, 2)
        if state in ("chase", "damage"):
            pygame.draw.rect(surface, accent, surface.get_rect(), 1)
        self._sprite_cache[key] = surface
        return surface

    def _update_particles(self, dt: float) -> None:
        alive: list[Particle] = []
        for particle in self.particles:
            particle.position += particle.velocity * dt
            particle.velocity *= 0.94
            particle.life -= dt
            if particle.life > 0.0:
                alive.append(particle)
        self.particles = alive

    def _update_presentation_state(self, dt: float) -> None:
        for caption in self.captions:
            caption.life -= dt
        self.captions = [caption for caption in self.captions if caption.life > 0.0]
        for effect in self.effects:
            effect.life -= dt
        self.effects = [effect for effect in self.effects if effect.life > 0.0]
        self.banner_life = max(0.0, self.banner_life - dt)
        self.security_clear_life = max(0.0, self.security_clear_life - dt)
        self.room_label_life = max(0.0, self.room_label_life - dt)
        tile = (int(self.sim.player[0] // TILE_SIZE), int(self.sim.player[1] // TILE_SIZE))
        role = self._room_role_at(*tile)
        if role != self._last_room_role:
            self._last_room_role = role
            self.room_label = f"FACILITY // {role.upper()}"
            self.room_label_life = 1.6

    def _draw_alert_indicators(self) -> None:
        """Communicate off-screen or occluded pressure by direction, never exact hidden state."""

        threats: list[tuple[np.ndarray, float, str, str]] = []
        for guard in self.sim.level.guards:
            pressure = 1.0 if guard.mode == GuardMode.CHASE else guard.awareness
            if self._visible_from_player(guard.position):
                continue
            audible = self.sim.player_guard_audible_estimate(guard)
            if audible is not None:
                estimate, confidence = audible
                grade = ("I", "II", "III")[int(guard.grade)]
                threats.append(
                    (
                        estimate,
                        max(pressure, confidence),
                        "sound",
                        f"STEPS {grade} / {guard.mode.name}",
                    )
                )
            elif pressure > 0.15:
                memory = self.sim.security_intel.get(("guard", guard.guard_id))
                if memory is not None:
                    threats.append((memory.position, pressure, "guard", "GUARD"))
        for camera in self.sim.level.cameras:
            if camera.awareness > 0.15 and not self._visible_from_player(camera.position):
                threats.append((camera.position, camera.awareness, "camera", "CAM"))
        occupied_edges: list[np.ndarray] = []
        for position, pressure, kind, label in sorted(threats, key=lambda item: item[1], reverse=True):
            if len(occupied_edges) >= 3:
                break
            delta = position - self.sim.player
            direction = delta / max(1e-6, norm(delta))
            center = np.asarray((LOGICAL_SIZE[0] / 2, LOGICAL_SIZE[1] / 2))
            edge = center + direction * min(
                268.0 / max(abs(float(direction[0])), 0.001),
                132.0 / max(abs(float(direction[1])), 0.001),
            )
            if any(float(np.linalg.norm(edge - occupied)) < 28.0 for occupied in occupied_edges):
                continue
            occupied_edges.append(edge)
            tip = edge.astype(int)
            lateral = np.asarray((-direction[1], direction[0]))
            base = edge - direction * 10
            points = [tip, (base + lateral * 6).astype(int), (base - lateral * 6).astype(int)]
            color = self._vision_color(pressure)
            pygame.draw.polygon(self.logical, color, points, 2)
            glyph_center = tuple((base - direction * 7).astype(int))
            # The directional arrow is already the guard's triangle cue; only
            # electronic/sound sources need a second semantic glyph.
            if kind != "guard":
                self._draw_threat_glyph(self.logical, kind, glyph_center, color)
            label_width = self.font_small.size(label)[0]
            label_x = int(np.clip(tip[0] - label_width / 2, 4, LOGICAL_SIZE[0] - label_width - 4))
            label_y = int(np.clip(tip[1] + (9 if direction[1] <= 0.7 else -18), 68, 340))
            detection_left = (LOGICAL_SIZE[0] - 210) // 2
            if label_y < 91 and label_x + label_width >= detection_left and label_x <= detection_left + 210:
                label_y = 96
            self._text(label, label_x, label_y, self.font_small, color)

    def _draw_detection_status(self) -> None:
        """Show exposure as a readable acquire meter, without hidden positions."""

        records: list[tuple[float, str]] = [
            (camera.awareness, "camera") for camera in self.sim.level.cameras if camera.awareness > 0.01
        ]
        records.extend((guard.awareness, "guard") for guard in self.sim.level.guards if guard.awareness > 0.01)
        chasing = [guard for guard in self.sim.level.guards if guard.mode == GuardMode.CHASE]
        searching = [
            guard
            for guard in self.sim.level.guards
            if guard.mode in (GuardMode.INVESTIGATE, GuardMode.SEARCH)
        ]
        if not records and not chasing and not searching and self.security_clear_life <= 0.0:
            return
        pressure, kind = max(records, default=(0.0, "guard"), key=lambda item: item[0])
        confirmed = pressure >= 0.995 or bool(chasing)
        if chasing and pressure < 0.995:
            kind = "guard"
        color = GREEN if not records and not chasing and not searching else self._vision_color(1.0 if confirmed else pressure)
        width, height = 210, 18
        x, y = (LOGICAL_SIZE[0] - width) // 2, 70
        pygame.draw.rect(self.logical, (3, 8, 12), (x, y, width, height), border_radius=3)
        pygame.draw.rect(self.logical, color, (x, y, width, height), 1, border_radius=3)
        self._draw_threat_glyph(self.logical, kind, (x + 11, y + 9), color, filled=confirmed)

        if chasing:
            if self.sim._seen_by_guard:
                label = "SPOTTED // BREAK SIGHT"
            else:
                remaining = max(guard.mode_seconds for guard in chasing)
                label = f"LINE BROKEN // EVADE {remaining:3.1f}s"
            self._text(label, x + 22, y + 5, self.font_small, color)
            return
        if pressure >= 0.995:
            self._text("DETECTED // BREAK SIGHT", x + 22, y + 5, self.font_small, color)
            return

        if searching and pressure <= 0.01:
            remaining = max(guard.mode_seconds for guard in searching)
            self._text(f"SEARCHING // HOLD COVER {remaining:3.1f}s", x + 22, y + 5, self.font_small, AMBER)
            return

        if not records and self.security_clear_life > 0.0:
            self._text("CLEAR // PATROL RESET", x + 22, y + 5, self.font_small, GREEN)
            return

        source = "CAMERA" if kind == "camera" else "GUARD"
        self._text(f"{source} ACQUIRING", x + 22, y + 5, self.font_small, color)
        segments = 8
        filled_segments = max(1, int(math.ceil(pressure * segments)))
        for index in range(segments):
            pygame.draw.rect(
                self.logical,
                color if index < filled_segments else (39, 50, 54),
                (x + 150 + index * 7, y + 7, 5, 4),
            )

    def _draw_objective_indicator(self) -> None:
        if self.sim.quota_met:
            target = self.sim.level.extraction
            color = GREEN
            label = "EXIT"
        else:
            terminal = self.sim.objective_terminal()
            if terminal is None:
                return
            target = terminal.position
            color = AMBER
            label = f"DATA {terminal.value}"
        screen = np.asarray(self._world(target), dtype=np.float32)
        if 28 <= screen[0] <= 612 and 76 <= screen[1] <= 322:
            return
        delta = target - self.sim.player
        direction = delta / max(1e-6, norm(delta))
        center = np.asarray((320.0, 190.0))
        edge = center + direction * min(
            278.0 / max(abs(float(direction[0])), 0.001),
            123.0 / max(abs(float(direction[1])), 0.001),
        )
        x, y = int(edge[0]), int(edge[1])
        pygame.draw.polygon(self.logical, color, ((x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)), 2)
        distance_m = norm(delta) / TILE_SIZE
        self._text(f"{label} {distance_m:.0f}m", x - 24, y + 9, self.font_small, color)

    def _draw_particles(self) -> None:
        for particle in self.particles:
            sx, sy = self._world(particle.position)
            pygame.draw.rect(self.logical, particle.color, (sx, sy, particle.size, particle.size))

    def _draw_lighting(self) -> None:
        self.light_layer.fill((2, 5, 9, 58 if not self.sim.lockdown else 38))
        px, py = self._world(self.sim.player)
        for radius, alpha in ((150, 0), (205, 28), (260, 50)):
            pygame.draw.circle(self.light_layer, (0, 0, 0, alpha), (px, py), radius)
        if self.sim.lockdown:
            pulse = 19 if self.reduced_flashes else int(18 + 12 * (0.5 + 0.5 * math.sin(self._time * 7)))
            self.light_layer.fill((90, 0, 12, pulse), special_flags=pygame.BLEND_RGBA_ADD)
        self.logical.blit(self.light_layer, (0, 0))
        glow = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        for terminal in self.sim.level.terminals:
            if terminal.completed:
                continue
            tx, ty = self._world(terminal.position)
            pygame.draw.circle(glow, (245, 184, 76, 16), (tx, ty), 38)
            pygame.draw.circle(glow, (245, 184, 76, 22), (tx, ty), 18)
        if self.sim.quota_met:
            ex, ey = self._world(self.sim.level.extraction)
            pygame.draw.circle(glow, (87, 226, 139, 24), (ex, ey), 54)
        # Alpha-composite rather than add raw RGB; this keeps terminals luminous
        # without washing out a runner who is actively linking at the console.
        self.logical.blit(glow, (0, 0))

    def _draw_screen_effects(self) -> None:
        effect_layer = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        for effect in self.effects:
            progress = 1.0 - effect.life / effect.duration
            alpha = int(180 * max(0.0, 1.0 - progress))
            if effect.kind == "dash_noise":
                sx, sy = self._world(effect.position)
                radius = 185 if self.reduced_motion else int(10 + 175 * progress)
                pygame.draw.circle(effect_layer, (*CYAN, max(30, alpha // 2)), (sx, sy), radius, 1)
            elif effect.kind == "pulse":
                sx, sy = self._world(effect.position)
                radius = int(18 + 155 * progress)
                pygame.draw.circle(effect_layer, (*CYAN, alpha), (sx, sy), radius, 2)
                if not self.reduced_flashes:
                    pygame.draw.circle(effect_layer, (*CYAN, alpha // 4), (sx, sy), max(1, radius - 8), 3)
            elif effect.kind == "damage":
                vignette = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
                strength = min(95, alpha // 2)
                for inset in range(0, 28, 4):
                    pygame.draw.rect(vignette, (RED[0], RED[1], RED[2], max(0, strength - inset * 2)), (inset, inset, 640 - inset * 2, 360 - inset * 2), 4)
                self.logical.blit(vignette, (0, 0))
            elif effect.kind == "quota_met":
                pygame.draw.rect(effect_layer, (*GREEN, min(36, alpha // 5)), (0, 0, 640, 360))
        self.logical.blit(effect_layer, (0, 0))
        if self.banner_life > 0.0:
            width = self.font.size(self.banner_text)[0] + 36
            y = 88
            pygame.draw.rect(self.logical, (4, 10, 14), (320 - width // 2, y, width, 28), border_radius=3)
            pygame.draw.rect(self.logical, CYAN if not self.sim.lockdown else RED, (320 - width // 2, y, width, 28), 1, border_radius=3)
            self._text(self.banner_text, 320 - width // 2 + 18, y + 7, self.font, INK)
        elif self.room_label_life > 0.0:
            width = self.font_small.size(self.room_label)[0] + 20
            # Detection owns y=70..88. The old room tag sat directly beneath
            # that rectangle and its delayed native glyphs visibly collided.
            pygame.draw.rect(self.logical, (4, 10, 14), (320 - width // 2, 96, width, 18), border_radius=2)
            pygame.draw.rect(self.logical, MUTED, (320 - width // 2, 96, width, 18), 1, border_radius=2)
            self._text(self.room_label, 320 - width // 2 + 10, 100, self.font_small, MUTED)

    def _draw_captions(self) -> None:
        if not self.sound_captions_enabled or not self.captions:
            return
        y = 273 - (len(self.captions) - 1) * 18
        for caption in self.captions:
            width = min(300, self.font_small.size(caption.text)[0] + 14)
            x = 320 - width // 2
            pygame.draw.rect(self.logical, (3, 7, 10), (x, y, width, 16), border_radius=2)
            pygame.draw.rect(self.logical, caption.color, (x, y, 2, 16))
            self._text(caption.text, x + 8, y + 3, self.font_small, INK)
            y += 18

    def _apply_accessibility_filter(self) -> None:
        if not (self.high_contrast or self.color_safe):
            return
        pixels = pygame.surfarray.pixels3d(self.logical)
        if self.color_safe:
            red_mask = (pixels[:, :, 0] > 170) & (pixels[:, :, 0] > pixels[:, :, 1] * 1.35) & (pixels[:, :, 2] < 150)
            green_mask = (pixels[:, :, 1] > 140) & (pixels[:, :, 1] > pixels[:, :, 0] * 1.35)
            pixels[red_mask] = (255, 92, 190)
            pixels[green_mask] = (82, 184, 255)
        if self.high_contrast:
            lookup = np.clip((np.arange(256, dtype=np.float32) - 96.0) * 1.22 + 96.0, 0, 255).astype(np.uint8)
            pixels[:] = lookup[pixels]
        del pixels

    def _draw_hud(self, lab_stats: dict[str, Any] | None) -> None:
        hud_small, hud_font = self._hud_fonts[self.hud_scale]
        expanded = self.hud_scale > 1.0
        panel_width = 350 if expanded else 276
        panel_height = 58 if expanded else 49
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((5, 10, 14, 220))
        self.logical.blit(panel, (10, 9))
        self._text(f"T{self.sim.tier}  {TIERS[self.sim.tier].name.upper()}", 18, 14, hud_small, CYAN)
        self._text(f"DATA {self.sim.data}/{self.sim.level.quota}", 18, 30 if expanded else 29, hud_font, AMBER if not self.sim.quota_met else GREEN)
        int_x = 128 if expanded else 106
        trace_x = 228 if expanded else 172
        trace_width = 120 if expanded else 102
        self._pips(int_x, 38 if expanded else 33, self.sim.integrity, 3, GREEN, label="INTEGRITY", font=hud_small)
        self._bar(trace_x, 22, trace_width, 7, self.sim.trace / TRACE_MAX, RED if self.sim.trace > 70 else AMBER, "TRACE", font=hud_small)
        for threshold in (25.0, 50.0, 75.0):
            tick_x = trace_x + int(round(trace_width * threshold / TRACE_MAX))
            pygame.draw.line(self.logical, (235, 237, 221), (tick_x, 21), (tick_x, 30), 1)
        if self.sim.level.response_drones and self.sim.level.drone_trace_threshold <= TRACE_MAX:
            drone_x = trace_x + int(round(trace_width * self.sim.level.drone_trace_threshold / TRACE_MAX))
            pygame.draw.polygon(self.logical, VIOLET, ((drone_x, 19), (drone_x - 3, 15), (drone_x + 3, 15)))
        self._bar(trace_x, 48 if expanded else 42, trace_width, 5, self.sim.dash_energy / 100.0, CYAN, "DASH", font=hud_small)
        seconds = int(math.ceil(self.sim.remaining_seconds))
        clock_x = 379 if expanded else 295
        clock_color = RED if seconds < 25 else INK
        if self.timer_warnings and seconds < 30:
            pygame.draw.polygon(self.logical, clock_color, ((clock_x - 11, 18), (clock_x - 3, 18), (clock_x - 7, 10)), 2)
            pygame.draw.rect(self.logical, clock_color, (clock_x - 12, 8, 92, 52), 1, border_radius=3)
        self._text(f"{seconds // 60:02d}:{seconds % 60:02d}", clock_x, 15, self.font_large, clock_color)
        self._text(f"PULSE {self.sim.pulse_charges}", clock_x + 3, 42, hud_small, CYAN if self.sim.pulse_charges else MUTED)
        alert_text = ("CLEAR", "WATCH", "ALERT", "HUNT", "LOCKDOWN")[self.sim.alert_tier]
        self._text(alert_text, clock_x + 62, 44, self.font_small, RED if self.sim.alert_tier >= 2 else MUTED)

        if self.sim.active_hack_progress > 0.0:
            self._bar(225, 324, 190, 8, self.sim.active_hack_progress, AMBER, "LINKING")
        elif self.tutorial_hints and self.sim.context_hint:
            hint = self.sim.context_hint
            width = self.font_small.size(hint)[0] + 16
            pygame.draw.rect(self.logical, (6, 12, 17), (320 - width // 2, 326, width, 18), border_radius=3)
            pygame.draw.rect(self.logical, (38, 78, 82), (320 - width // 2, 326, width, 18), 1, border_radius=3)
            self._text(hint, 320 - width // 2 + 8, 331, self.font_small, CYAN if self.sim.quota_met else (190, 208, 207))
        self._draw_minimap()
        if lab_stats:
            # Live inference should be observable without becoming a second
            # HUD. The former 230x104 panel covered the lower-left route and
            # terminals; this compact card uses under four percent of the
            # playfield and keeps full run details for the debrief/web shell.
            panel = pygame.Surface((218, 44), pygame.SRCALPHA)
            panel.fill((5, 10, 14, 208))
            self.logical.blit(panel, (10, 70))
            pygame.draw.rect(self.logical, VIOLET, (10, 70, 3, 44))
            policy_name = str(lab_stats.get("policy", "SCRIPTED BASELINE"))
            if len(policy_name) > 23:
                policy_name = policy_name.replace(" POLICY", "")[:23]
            self._text(f"AGENT // {policy_name}", 19, 76, self.font_small, VIOLET)
            action_text = str(lab_stats.get("action", "HOLD"))
            latency = float(lab_stats.get("latency_ms", 0.0))
            phase = "EXTRACT" if self.sim.quota_met else "ACQUIRE DATA"
            self._text(f"{action_text:<11} {latency:4.1f}ms", 19, 91, self.font_small, CYAN)
            self._text(f"{phase}  T{self.sim.trace:02.0f}", 19, 103, self.font_small, GREEN if self.sim.quota_met else AMBER)

    def _draw_minimap(self) -> None:
        width, height = 100, 56
        x0, y0 = 529, 9
        pygame.draw.rect(self.logical, (5, 10, 14), (x0, y0, width, height), border_radius=2)
        sx = (width - 6) / self.sim.level.grid.shape[1]
        sy = (height - 6) / self.sim.level.grid.shape[0]
        for room in self.sim.level.rooms:
            rect = pygame.Rect(x0 + 3 + room.x * sx, y0 + 3 + room.y * sy, max(1, room.width * sx), max(1, room.height * sy))
            pygame.draw.rect(self.logical, ROLE_COLORS.get(room.role, (40, 45, 50)), rect)
        px = x0 + 3 + self.sim.player[0] / TILE_SIZE * sx
        py = y0 + 3 + self.sim.player[1] / TILE_SIZE * sy
        pygame.draw.circle(self.logical, CYAN, (int(px), int(py)), 2)
        ex = x0 + 3 + self.sim.level.extraction[0] / TILE_SIZE * sx
        ey = y0 + 3 + self.sim.level.extraction[1] / TILE_SIZE * sy
        pygame.draw.circle(self.logical, GREEN if self.sim.quota_met else MUTED, (int(ex), int(ey)), 2, 1)
        for terminal in self.sim.level.terminals:
            if terminal.completed:
                continue
            tx = x0 + 3 + terminal.position[0] / TILE_SIZE * sx
            ty = y0 + 3 + terminal.position[1] / TILE_SIZE * sy
            pygame.draw.rect(self.logical, AMBER, (int(tx), int(ty), 2, 2))
        for key, memory in self._security_memory.items():
            mx = x0 + 3 + memory.position[0] / TILE_SIZE * sx
            my = y0 + 3 + memory.position[1] / TILE_SIZE * sy
            color = AMBER if memory.kind == "camera" else (VIOLET if memory.kind == "drone" else RED)
            if memory.kind == "camera":
                pygame.draw.rect(self.logical, color, (int(mx) - 1, int(my) - 1, 3, 3), 1)
            else:
                pygame.draw.circle(self.logical, color, (int(mx), int(my)), 1)

    def _bar(self, x: int, y: int, width: int, height: int, value: float, color: tuple[int, int, int], label: str, *, font: pygame.font.Font | None = None) -> None:
        self._text(label, x, y - 10, font or self.font_small, MUTED)
        pygame.draw.rect(self.logical, (19, 29, 34), (x, y, width, height))
        pygame.draw.rect(self.logical, color, (x, y, int(width * np.clip(value, 0.0, 1.0)), height))

    def _pips(self, x: int, y: int, value: int, maximum: int, color: tuple[int, int, int], *, label: str, font: pygame.font.Font | None = None) -> None:
        self._text(label, x, y - 14, font or self.font_small, MUTED)
        for index in range(maximum):
            pygame.draw.rect(self.logical, color if index < value else (32, 45, 49), (x + index * 15, y, 11, 7), border_radius=2)

    def _draw_menu_backdrop(self, title: str = "") -> None:
        # The opening screen deliberately uses the same flat 2D visual language
        # as gameplay. Earlier illustrated pseudo-isometric key art suggested a
        # different game and was removed after hands-on review.
        gradient = pygame.Surface(LOGICAL_SIZE)
        for y in range(LOGICAL_SIZE[1]):
            t = y / LOGICAL_SIZE[1]
            pygame.draw.line(gradient, (int(7 + 5 * t), int(12 + 10 * t), int(18 + 13 * t)), (0, y), (640, y))
        self.logical.blit(gradient, (0, 0))
        for y in range(0, 360, 18):
            pygame.draw.line(self.logical, (10, 20, 25), (0, y), (640, y))
        for x in range(0, 640, 18):
            pygame.draw.line(self.logical, (10, 20, 25), (x, 0), (x, 360))
        pygame.draw.polygon(self.logical, (10, 27, 33), ((344, 0), (640, 0), (640, 360), (494, 360)))
        pygame.draw.polygon(self.logical, (17, 38, 43), ((388, 47), (613, 24), (625, 286), (444, 326)))
        for index in range(12):
            x = 403 + (index % 3) * 69
            y = 53 + (index // 3) * 61
            room = pygame.Rect(x, y, 50, 40)
            pygame.draw.rect(self.logical, (18, 38, 44), room, border_radius=2)
            pygame.draw.rect(self.logical, (49, 89, 92), room, 1, border_radius=2)
            pygame.draw.rect(self.logical, (27, 50, 55), room.inflate(-8, -8))
            if index % 3 == 0:
                pygame.draw.line(self.logical, VIOLET, (x + 8, y + 20), (x + 42, y + 20), 2)
            elif index % 3 == 1:
                pygame.draw.circle(self.logical, AMBER, room.center, 5, 1)
            else:
                pygame.draw.rect(self.logical, CYAN, (x + 13, y + 13, 24, 2))
        # Animated surveillance sweep and infiltrator silhouette.
        sweep_angle = -0.4 if self.reduced_motion else -0.8 + 0.45 * math.sin(self._time * 0.65)
        origin = np.asarray((595.0, 52.0))
        cone = [origin]
        for offset in (-0.34, 0.34):
            cone.append(origin + angle_vector(sweep_angle + math.pi / 2 + offset) * 245)
        overlay = pygame.Surface(LOGICAL_SIZE, pygame.SRCALPHA)
        pygame.draw.polygon(overlay, (245, 184, 76, 18), cone)
        pygame.draw.line(overlay, (245, 184, 76, 72), cone[0], cone[1])
        pygame.draw.line(overlay, (245, 184, 76, 72), cone[0], cone[2])
        self.logical.blit(overlay, (0, 0))
        runner = self._actor_sprite("runner", -0.7, True, "normal")
        runner = pygame.transform.scale(runner, (52, 60))
        runner.set_alpha(205)
        self.logical.blit(runner, (505, 246))
        pygame.draw.line(self.logical, CYAN, (469, 314), (593, 288), 2)
        for index in range(6):
            phase = 0 if self.reduced_motion else int(self._time * 18)
            x = 472 + ((index * 23 + phase) % 110)
            pygame.draw.rect(self.logical, CYAN if index % 2 else VIOLET, (x, 311 - (index % 3), 5, 1))
        pygame.draw.line(self.logical, CYAN, (0, 18), (640, 18), 2)
        self._text("PROCEDURAL INFILTRATION", 452, 337, self.font_small, MUTED)

    def _draw_native_text(self, target: pygame.Surface, scaled_size: tuple[int, int]) -> None:
        """Composite recorded text at output resolution instead of scaling it."""

        if not self._native_text_commands:
            return
        scale_x = scaled_size[0] / LOGICAL_SIZE[0]
        scale_y = scaled_size[1] / LOGICAL_SIZE[1]
        font_scale = min(scale_x, scale_y)
        for command in self._native_text_commands:
            size = max(7, int(round(command.design_size * font_scale)))
            font = self._native_font_cache.get(size)
            if font is None:
                font = pygame.font.SysFont("consolas", size, bold=True)
                self._native_font_cache[size] = font
            color = command.color
            if self.color_safe:
                if color == RED:
                    color = (255, 92, 190)
                elif color == GREEN:
                    color = (82, 184, 255)
            glyph = font.render(command.text, True, color)
            target.blit(glyph, (int(round(command.x * scale_x)), int(round(command.y * scale_y))))

    def _text(self, text: str, x: float, y: float, font: pygame.font.Font, color: tuple[int, int, int]) -> None:
        value = str(text)
        if self.visible:
            design_size = self._font_design_sizes.get(id(font), max(7, font.get_height() - 2))
            self._native_text_commands.append(NativeTextCommand(value, x, y, design_size, color))
            return
        self.logical.blit(font.render(value, False, color), (int(x), int(y)))

    @staticmethod
    def _wrap_text(text: str, font: pygame.font.Font, maximum_width: int) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= maximum_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines
