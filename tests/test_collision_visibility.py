from __future__ import annotations

import numpy as np
import pytest

from ghostline.config import TILE_SIZE
from ghostline.generation import PROP_VARIANTS, LevelGenerator
from ghostline.simulation import GhostlineSimulation, world_to_tile
from ghostline.types import Tile


@pytest.mark.parametrize(
    ("kind", "width", "height"),
    (
        ("server", 1, 4),
        ("server", 3, 1),
        ("locker", 1, 3),
        ("locker", 3, 1),
        ("sofa", 1, 3),
        ("console", 4, 1),
    ),
)
def test_modular_fixtures_preserve_exact_blocked_footprint(kind: str, width: int, height: int) -> None:
    tile_x, tile_y = 17, 23
    props = LevelGenerator._expand_visible_fixture(kind, tile_x, tile_y, width, height)
    original = {
        (tile_x + dx, tile_y + dy)
        for dy in range(height)
        for dx in range(width)
    }
    expanded = {
        (prop.tile_x + dx, prop.tile_y + dy)
        for prop in props
        for dy in range(prop.height)
        for dx in range(prop.width)
    }

    assert expanded == original
    assert len(props) == width * height
    assert all((prop.width, prop.height, prop.blocking) == (1, 1, True) for prop in props)


def test_every_authored_blocking_prop_tile_has_rendered_pixel_coverage(monkeypatch) -> None:
    """Catch collision cells that are blank after atlas bottom-pivoting.

    This is a pixel-level contract rather than a kind allow-list: every
    blocking cell in every authored furniture shape must contain a meaningful
    amount of non-floor prop art when rendered through the shipping path.
    """

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import BG, GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=10_101, tier=6), visible=False)
    checked: set[tuple[str, int, int]] = set()
    failures: list[tuple[str, int, int, int, int, int]] = []
    try:
        for variants in PROP_VARIANTS.values():
            for variant in variants:
                for kind, _, _, width, height in variant:
                    props = LevelGenerator._expand_visible_fixture(kind, 10, 10, width, height)
                    for prop in props:
                        identity = (prop.kind, prop.width, prop.height)
                        if identity in checked:
                            continue
                        checked.add(identity)
                        renderer.logical.fill(BG)
                        screen_x, screen_y = 320, 180
                        pixel_width = prop.width * TILE_SIZE - 6
                        pixel_height = prop.height * TILE_SIZE - 6
                        rect = pygame.Rect(
                            screen_x - pixel_width // 2,
                            screen_y - pixel_height // 2,
                            pixel_width,
                            pixel_height,
                        )
                        renderer._draw_prop(prop, rect)
                        pixels = pygame.surfarray.array3d(renderer.logical)
                        for tile_y in range(prop.height):
                            for tile_x in range(prop.width):
                                left = screen_x - prop.width * TILE_SIZE // 2 + tile_x * TILE_SIZE
                                top = screen_y - prop.height * TILE_SIZE // 2 + tile_y * TILE_SIZE
                                cell = pixels[left : left + TILE_SIZE, top : top + TILE_SIZE]
                                coverage = int(
                                    np.count_nonzero(
                                        np.any(cell != np.asarray(BG, dtype=np.uint8), axis=2)
                                    )
                                )
                                if coverage < 100:
                                    failures.append(
                                        (prop.kind, prop.width, prop.height, tile_x, tile_y, coverage)
                                    )
    finally:
        renderer.close()

    assert checked
    assert not failures, f"blocking cells without readable prop art: {failures}"


def test_doors_and_security_do_not_create_unrepresented_collision() -> None:
    """Security actors are hazards, while doors remain visibly walkable tiles."""

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    for door in sim.level.doors:
        x, y = door.tile
        assert sim.level.grid[y, x] == Tile.DOOR
        assert door.tile not in sim._blocked_tiles
    for camera in sim.level.cameras:
        assert world_to_tile(camera.position) not in sim._blocked_tiles
    for guard in sim.level.guards:
        assert world_to_tile(guard.position) not in sim._blocked_tiles

