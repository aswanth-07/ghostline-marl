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
    # The authored world remains pixel art, but a phone normally downsamples
    # the 1280x720 browser framebuffer into a much smaller CSS viewport.
    # Nearest-neighbour downsampling exaggerates block edges and makes the
    # native-resolution HUD harder to read, so touch/small screens use the
    # browser's high-quality compositor. Desktop retains crisp integer pixels.
    smooth_phone_output = bool(
        platform.window.matchMedia("(pointer: coarse)").matches
        or int(platform.window.innerWidth) <= 900
    )
    platform.window.canvas.style.imageRendering = "auto" if smooth_phone_output else "pixelated"
    platform.window.canvas.dataset.scaling = "smooth-mobile" if smooth_phone_output else "crisp-desktop"
    platform.window.canvas.setAttribute("aria-label", "Ghostline stealth game")
    hydrate_progression(platform.window, progression_path())
    await GhostlineWebRuntime(GameApp(mode="menu")).run()


asyncio.run(main())
