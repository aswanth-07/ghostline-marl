"""Render the shipping eight-direction locomotion matrix at logical scale."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from ghostline.presentation import BG, CYAN, RED, GhostlineRenderer
from ghostline.simulation import GhostlineSimulation
from ghostline.types import GuardMode


def render(output: Path) -> None:
    renderer = GhostlineRenderer(GhostlineSimulation(seed=2_000_004, tier=6), visible=False)
    sheet = pygame.Surface((640, 360))
    sheet.fill(BG)
    title = pygame.font.SysFont("consolas", 16, bold=True)
    label = pygame.font.SysFont("consolas", 10, bold=True)
    sheet.blit(title.render("GHOSTLINE // EIGHT-DIRECTION LOCOMOTION QA", False, CYAN), (16, 12))
    directions = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
    for index, name in enumerate(directions):
        sheet.blit(label.render(name, False, (157, 177, 184)), (54 + index * 72, 40))

    for phase in range(4):
        renderer._time = phase / 10.0 + 0.01
        for direction in range(8):
            angle = direction * math.tau / 8.0
            x = 58 + direction * 72
            runner = renderer._runner_atlas_sprite(angle, True, "normal")
            guard = renderer._guard_atlas_sprite(angle, GuardMode.PATROL, True, 0.0)
            if runner is not None:
                sheet.blit(runner, (x - runner.get_width() // 2, 61 + phase * 39))
            if guard is not None:
                sheet.blit(guard, (x - guard.get_width() // 2, 216 + phase * 34))
        sheet.blit(label.render(f"F{phase + 1}", False, CYAN), (16, 71 + phase * 39))
        sheet.blit(label.render(f"F{phase + 1}", False, RED), (16, 225 + phase * 34))

    pygame.draw.line(sheet, (32, 61, 72), (12, 204), (628, 204), 1)
    sheet.blit(label.render("RUNNER", False, CYAN), (566, 188))
    sheet.blit(label.render("GUARD", False, RED), (572, 344))
    output.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(sheet, output)
    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visual-qa/locomotion-v2/eight-direction-logical.png"),
    )
    args = parser.parse_args()
    render(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
