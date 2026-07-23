"""Render and verify Ghostline's release-scale presentation matrix.

This is deliberately a presentation-only tool. It freezes representative
logical frames and presents them through the shipping ``GhostlineRenderer``
at 2x and 3x. The world remains an exact nearest-neighbour expansion of the
640x360 source, while a deliberate pixel difference proves that text was
rerasterized at native output resolution instead of enlarging old glyphs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Callable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from ghostline.presentation import GhostlineRenderer
from ghostline.simulation import GhostlineSimulation
from ghostline.types import Drone, GuardMode


LOGICAL_SIZE = (640, 360)
RELEASE_SIZES = ((1280, 720), (1920, 1080))


def _screen_array(surface: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()


def _visible_security_point(
    sim: GhostlineSimulation,
    preferred_angle: float,
    occupied: list[np.ndarray],
) -> np.ndarray:
    """Find a valid staged actor position whose complete cone is inspectable."""

    fallback: np.ndarray | None = None
    fallback_separation = -1.0
    for radius in (112.0, 92.0, 132.0, 72.0):
        for offset in (0.0, 0.28, -0.28, 0.56, -0.56, 0.9, -0.9, math.pi):
            angle = preferred_angle + offset
            candidate = sim.player + np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32) * radius
            if not sim._can_occupy(candidate, 9.0) or not sim.line_of_sight(sim.player, candidate):
                continue
            separation = min((float(np.linalg.norm(candidate - other)) for other in occupied), default=999.0)
            if separation > fallback_separation:
                fallback, fallback_separation = candidate, separation
            if separation < 68.0:
                continue
            occupied.append(candidate)
            return candidate
    # Deterministic valid fallback for unusually crowded authored terminals.
    candidate = fallback if fallback is not None else sim.player.copy()
    occupied.append(candidate)
    return candidate


def _configure_gameplay(sim: GhostlineSimulation) -> None:
    """Stage a deterministic, non-mutating-in-runtime security showcase."""

    sim.explored[:] = True
    sim.elapsed_ticks = 3_060
    sim.trace = 71.0
    sim.trace_floor = 36.0
    sim.max_trace = 71.0
    sim.player = sim.level.terminals[0].position.copy()
    sim.heading[:] = (1.0, 0.0)
    sim.level.terminals[0].progress = sim.level.terminals[0].hack_seconds * 0.62
    sim._active_hack = 0  # QA staging only; never enters simulation/runtime code.
    occupied: list[np.ndarray] = []
    camera_position = _visible_security_point(sim, -1.9, occupied) if sim.level.cameras else None
    first_guard_position = _visible_security_point(sim, 0.35, occupied) if sim.level.guards else None
    second_guard_position = _visible_security_point(sim, 2.65, occupied) if len(sim.level.guards) > 1 else None
    drone_position = _visible_security_point(sim, 1.75, occupied)

    if sim.level.guards:
        guard = sim.level.guards[0]
        guard.position = first_guard_position
        guard.mode = GuardMode.CHASE
        guard.awareness = 0.87
        guard.facing = float(np.arctan2(sim.player[1] - guard.position[1], sim.player[0] - guard.position[0]))
        guard.attack_windup = 0.28
    if len(sim.level.guards) > 1:
        guard = sim.level.guards[1]
        guard.position = second_guard_position
        guard.mode = GuardMode.SUSPICIOUS
        guard.awareness = 0.46
        guard.facing = float(np.arctan2(sim.player[1] - guard.position[1], sim.player[0] - guard.position[0]))
    if sim.level.cameras:
        camera = sim.level.cameras[0]
        camera.position = camera_position
        camera.angle = float(np.arctan2(sim.player[1] - camera.position[1], sim.player[0] - camera.position[0]))
        camera.detected = False
        camera.awareness = 0.62
    sim.drones = [
        Drone(
            drone_id=9001,
            position=drone_position,
            facing=2.55,
            attack_windup=0.34,
        )
    ]
    if hasattr(sim, "security_doors"):
        from ghostline.types_v3 import Decoy, ShockProjectile

        if sim.security_doors:
            sim.security_doors[0].lock_remaining = 3.0
            sim._refresh_navigation_blocks()
        decoy_position = sim.player + np.asarray((-48.0, 36.0), dtype=np.float32)
        if not sim._can_occupy(decoy_position, 5.0):
            decoy_position = sim.player.copy()
        sim.decoys = [Decoy(9002, decoy_position, 1.4, 0.2)]
        projectile_position = sim.player + np.asarray((74.0, -18.0), dtype=np.float32)
        sim.projectiles = [
            ShockProjectile(
                9003,
                projectile_position,
                np.asarray((-220.0, 28.0), dtype=np.float32),
                sim.level.guards[-1].guard_id,
                0.8,
            )
        ]
        suppressor = sim.operative_states[sim.level.guards[-1].guard_id]
        suppressor.aim_progress = 0.52
        suppressor.aim_target = sim.player.copy()


def _title(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="GHOSTLINE",
        subtitle="Move unseen. Take the signal. Leave no trace.",
        items=["PLAY CONTRACTS", "AGENT LAB", "HOW TO PLAY", "SETTINGS", "CREDITS", "QUIT"],
        selected=0,
        badge="PROCEDURAL STEALTH // RL SHOWCASE",
        panel=[
            "OPERATIVE STATUS",
            "CLEARANCE       6/6",
            "CONTRACTS WON   6/6",
            "RUNTIME POLICY  RECURRENT ONNX",
            "",
            "ONE WORLD. TWO CONTROLLERS.",
            "IDENTICAL SENSORS AND RULES.",
        ],
        footer="W/S  NAVIGATE     ENTER  SELECT",
        return_array=True,
    )


def _settings(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="SETTINGS",
        subtitle="Saved instantly to your local Ghostline profile.",
        items=["AUDIO MIX", "ACCESSIBILITY", "CONTROLS", "DISPLAY", "BACK"],
        selected=1,
        panel=[
            "ACCESS PROFILE",
            "CAPTIONS       ON",
            "HIGH CONTRAST  OFF",
            "COLOR-SAFE     OFF",
            "HUD SCALE      100%",
            "",
            "Every gameplay key is remappable.",
        ],
        footer="ENTER  OPEN     ESC  BACK",
        return_array=True,
    )


def _briefing(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="TIER 6: GHOSTLINE",
        subtitle="The full system is awake.",
        body=[
            "Take the highest-value route you can survive, satisfy the contract, and disappear.",
            "",
            "FACILITY  5x3 MODULES",
            "CONTRACT QUOTA  8 DATA",
            "SECURITY  5 CAMERAS / 6 GUARDS",
            "WINDOW  3:45",
        ],
        badge="CONTRACT // 06",
        panel=[
            "FIELD PROTOCOL",
            "AMBER   DATA TERMINAL",
            "GREEN   EXTRACTION RELAY",
            "CONE    ACTIVE SIGHTLINE",
            "I/II/III  PATROL GRADE",
            "TRACE   NETWORK PRESSURE",
            "",
            "SEED NAMESPACE  PROCEDURAL",
        ],
        footer="ENTER  DEPLOY     ESC  CONTRACTS",
        return_array=True,
    )


def _field_manual(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="FIELD MANUAL",
        subtitle="A contract is information, pressure, and an exit.",
        body=[
            "W/A/S/D   MOVE",
            "SHIFT       DASH // FAST, LOUD, ENERGY-LIMITED",
            "SPACE       DISRUPTION PULSE // LIMITED CHARGES",
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
            "Segments fill before detection; walls reset pressure.",
            "Pulse disables electronics; dash makes noise.",
        ],
        footer="ESC OR ENTER  BACK",
        return_array=True,
    )


def _pause(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="CONTRACT PAUSED",
        subtitle="Simulation held at a deterministic tick boundary.",
        items=["RESUME", "RESTART CONTRACT", "FIELD MANUAL", "SETTINGS", "ABORT TO MENU"],
        selected=0,
        panel=[
            "LIVE CONTRACT",
            "TIER 6 // GHOSTLINE",
            "DATA 0/8",
            "INTEGRITY 3/3",
            "TRACE 71.0",
            "",
            "No mission time advances while paused.",
        ],
        footer="ENTER  SELECT     ESC  RESUME",
        return_array=True,
    )


def _debrief(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="CONTRACT CLEARED",
        subtitle="Ghostline established. No recoverable signal remains.",
        body=[
            "SCORE       018420",
            "DATA        8 / 8",
            "OPTIONAL    2",
            "DURATION    02:54",
            "MAX TRACE   71.0",
            "DETECTIONS  1",
            "DAMAGE      0",
            "EFFICIENCY  82%",
        ],
        compact_body=True,
        badge="OPERATIVE RUN",
        panel=[
            "BENCHMARK RECORD",
            "CONTROLLER  HUMAN",
            "DISTANCE    2218.4px",
            "IDLE          2.1%",
            "PULSE USE   2",
            "POLICY P95  —",
        ],
        footer="ENTER  NEXT CONTRACT     R  REPLAY SEED     ESC  MENU",
        return_array=True,
    )


def _accessibility(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="ACCESSIBILITY",
        subtitle="Danger always uses color plus shape and text.",
        items=[
            "HIGH CONTRAST   OFF",
            "COLOR-SAFE CUES ON",
            "REDUCED MOTION  ON",
            "REDUCED FLASHES ON",
            "SOUND CAPTIONS  ON",
            "HUD SCALE       125%",
            "TIMER ASSIST    ON",
            "TIMER WARNINGS  ON",
            "TUTORIAL HINTS  ON",
            "BACK",
        ],
        selected=6,
        compact=True,
        panel=[
            "TIMER ASSIST adds 35% to human contract windows and is recorded in telemetry.",
            "REDUCED MOTION disables shake, trails, and moving UI art.",
            "COLOR-SAFE remaps danger to pink and extraction to blue.",
            "CAPTIONS identify alerts, impacts, pulses, and terminals.",
        ],
        footer="ENTER  TOGGLE     ESC  BACK",
        return_array=True,
    )


def _agent_lab(renderer: GhostlineRenderer) -> None:
    renderer.draw_screen(
        title="AGENT LAB",
        subtitle="Player-equivalent // deterministic replay",
        items=[
            "WATCH TIER 1  ORIENTATION",
            "WATCH TIER 2  SURVEILLANCE",
            "WATCH TIER 3  PATROL",
            "WATCH TIER 4  COUNTERMEASURE",
            "WATCH TIER 5  LOCKDOWN",
            "WATCH TIER 6  GHOSTLINE",
        ],
        selected=5,
        badge="RECURRENT ONNX POLICY",
        panel=[
            "EVALUATION SEED  2000123",
            "PUBLIC SENSORS // NO HIDDEN STATE",
            "LEFT/RIGHT +/-1",
            "SHIFT + LEFT/RIGHT +/-100",
            "",
            "MATCHED LOCAL RESULTS",
            "HUMAN   CLEAR 112.4s T58.1",
            "AGENT   CLEAR  71.8s T31.6",
        ],
        footer="ENTER  WATCH     LEFT/RIGHT  SEED     ESC  BACK",
        return_array=True,
    )


def _gameplay(renderer: GhostlineRenderer) -> None:
    renderer.apply_accessibility(
        {
            "color_safe": True,
            "high_contrast": False,
            "reduced_motion": True,
            "reduced_flashes": True,
            "sound_captions": True,
            "hud_scale": 1.25,
            "timer_warnings": True,
            "tutorial_hints": True,
        }
    )
    renderer.camera = renderer.sim.player.copy()
    renderer.draw(return_array=True)


def _agent_lab_live(renderer: GhostlineRenderer) -> None:
    renderer.apply_accessibility(
        {
            "color_safe": False,
            "high_contrast": True,
            "reduced_motion": True,
            "reduced_flashes": True,
            "sound_captions": True,
            "hud_scale": 1.0,
            "timer_warnings": True,
            "tutorial_hints": True,
        }
    )
    renderer.camera = renderer.sim.player.copy()
    renderer.draw(
        return_array=True,
        lab_stats={
            "policy": "RECURRENT ONNX POLICY",
            "action": 16,
            "latency_ms": 1.17,
            "hidden_norm": 8.42,
            "objective": "ACQUIRE DATA",
            "seed": 2_000_123,
            "tier": 6,
        },
    )


SCENES: dict[str, Callable[[GhostlineRenderer], None]] = {
    "title": _title,
    "briefing": _briefing,
    "field-manual": _field_manual,
    "pause": _pause,
    "debrief": _debrief,
    "settings": _settings,
    "accessibility-timer-assist": _accessibility,
    "agent-lab": _agent_lab,
    "gameplay": _gameplay,
    "agent-lab-live": _agent_lab_live,
}


def _capture_scaled(
    renderer: GhostlineRenderer,
    *,
    output: Path,
    scene: str,
    size: tuple[int, int],
) -> dict[str, object]:
    scale = size[0] // LOGICAL_SIZE[0]
    logical = _screen_array(renderer.logical)
    renderer.window = pygame.display.set_mode(size)
    renderer._present(return_array=False)  # Exercise the shipping presentation path.
    actual = _screen_array(renderer.window)
    expected = np.repeat(np.repeat(logical, scale, axis=0), scale, axis=1)
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    native_text_pixels = int(np.count_nonzero(np.any(difference != 0, axis=2)))
    destination = output / f"{scene}-{size[0]}x{size[1]}.png"
    pygame.image.save(renderer.window, str(destination))
    return {
        "scene": scene,
        "size": list(size),
        "integer_scale": scale,
        "output": destination.as_posix(),
        "native_text_runs": len(renderer._native_text_commands),
        "native_text_pixels": native_text_pixels,
        "maximum_channel_error": int(difference.max(initial=0)),
        "cropped_or_letterboxed": bool(actual.shape != expected.shape),
    }


def _benchmark(renderer: GhostlineRenderer, size: tuple[int, int], frames: int) -> dict[str, float | int | list[int]]:
    renderer.window = pygame.display.set_mode(size)
    renderer.apply_accessibility({"reduced_motion": True, "reduced_flashes": True, "hud_scale": 1.0})
    renderer.camera = renderer.sim.player.copy()
    for _ in range(20):
        renderer.draw(return_array=False)
    start = time.perf_counter()
    for _ in range(frames):
        renderer.draw(return_array=False)
    full_seconds = time.perf_counter() - start

    present_frames = frames * 4
    start = time.perf_counter()
    for _ in range(present_frames):
        renderer._present(return_array=False)
    present_seconds = time.perf_counter() - start
    return {
        "size": list(size),
        "frames": frames,
        "full_render_present_ms": round(full_seconds * 1000.0 / frames, 4),
        "full_render_present_fps": round(frames / full_seconds, 2),
        "scale_present_ms": round(present_seconds * 1000.0 / present_frames, 4),
        "scale_present_fps": round(present_frames / present_seconds, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/visual-qa/scaled-final"))
    parser.add_argument("--benchmark-frames", type=int, default=180)
    parser.add_argument("--adaptive", action="store_true", help="stage Env-v3 doors, decoy, and projectile cues")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.adaptive:
        from ghostline.simulation_v3 import GhostlineSimulationV3

        sim = GhostlineSimulationV3(seed=2_000_123, tier=6, directive="ghost")
    else:
        sim = GhostlineSimulation(seed=2_000_123, tier=6)
    _configure_gameplay(sim)
    # SDL's dummy video backend still exercises the shipping visible renderer,
    # including its post-scale native typography pass.
    renderer = GhostlineRenderer(sim, visible=True)
    captures: list[dict[str, object]] = []
    try:
        for scene, draw_scene in SCENES.items():
            draw_scene(renderer)
            for size in RELEASE_SIZES:
                captures.append(_capture_scaled(renderer, output=args.output, scene=scene, size=size))
        benchmarks = [_benchmark(renderer, size, args.benchmark_frames) for size in RELEASE_SIZES]
    finally:
        renderer.close()

    result = {
        "logical_size": list(LOGICAL_SIZE),
        "release_sizes": [list(size) for size in RELEASE_SIZES],
        "capture_count": len(captures),
        "adaptive": bool(args.adaptive),
        "all_native_text_scaled": all(
            capture["native_text_runs"] > 0
            and capture["native_text_pixels"] > 0
            and not capture["cropped_or_letterboxed"]
            for capture in captures
        ),
        "captures": captures,
        "benchmarks": benchmarks,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["all_native_text_scaled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
