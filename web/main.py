# /// script
# dependencies = [
#   "numpy",
# ]
# ///
import asyncio
import platform

import numpy  # Declared explicitly so pygbag installs the browser wheel.
import pygame

from ghostline.app import GameApp
from ghostline.progression import progression_path
from web_runtime import GhostlineWebRuntime, hydrate_progression


async def main() -> None:
    del numpy, pygame
    platform.window.canvas.style.imageRendering = "pixelated"
    platform.window.canvas.setAttribute("aria-label", "Ghostline stealth game")
    hydrate_progression(platform.window, progression_path())
    await GhostlineWebRuntime(GameApp(mode="menu")).run()


asyncio.run(main())
