"""The renderer's frame caches must not change what the frame looks like.

Terrain and vision-cone fans are memoised so the browser build can hold a frame
budget. Both caches are only safe while their invalidation is right, and a stale
cache fails silently by drawing a previous level or a previous pose. These tests
pin the invalidation and bound the visual difference.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from ghostline.presentation import GhostlineRenderer
from ghostline.simulation import GhostlineSimulation
from ghostline.types import Action, Tile


@pytest.fixture(scope="module", autouse=True)
def _display() -> None:
    pygame.init()
    pygame.display.set_mode((1280, 720))


def _renderer(seed: int, tier: int) -> tuple[GhostlineSimulation, GhostlineRenderer]:
    sim = GhostlineSimulation()
    sim.reset(seed=seed, tier=tier)
    return sim, GhostlineRenderer(sim, visible=False)


def _frame(renderer: GhostlineRenderer) -> np.ndarray:
    renderer.draw()
    return pygame.surfarray.array3d(renderer.logical).copy()


class _NeverRetains(dict):
    """A mapping that always misses, giving an uncached baseline."""

    def __setitem__(self, key, value) -> None:  # noqa: D105
        return


def test_terrain_cache_is_rebuilt_when_the_level_changes() -> None:
    sim, renderer = _renderer(7, 4)
    renderer.draw()
    first_floor, first_walls = renderer._floor_cache, renderer._wall_cache
    first_identity = renderer._terrain_identity
    assert first_floor is not None and first_walls is not None

    # Same level: the surfaces must be re-used rather than repainted.
    renderer.draw()
    assert renderer._floor_cache is first_floor
    assert renderer._wall_cache is first_walls

    sim.reset(seed=99, tier=6)
    renderer.draw()
    assert renderer._terrain_identity != first_identity
    assert renderer._floor_cache is not first_floor
    assert renderer._wall_cache is not first_walls
    # The new surfaces must match the new grid, not the old one.
    rows, columns = sim.level.grid.shape
    assert renderer._floor_cache.get_size() == (columns * 32, rows * 32)


def test_terrain_blit_lands_on_the_same_pixels_as_a_repainted_cache() -> None:
    """A cached frame must equal one drawn with a cold cache, exactly."""

    _, renderer = _renderer(31337, 3)
    warm = _frame(renderer)
    # Drop the cache and redraw the identical frame from scratch.
    renderer._floor_cache = None
    renderer._wall_cache = None
    renderer._terrain_identity = None
    cold = _frame(renderer)
    assert np.array_equal(warm, cold)


def test_wall_layer_keeps_non_wall_pixels_transparent() -> None:
    """Walls composite after the vision cones, so the layer must not be opaque."""

    sim, renderer = _renderer(7, 4)
    renderer.draw()
    walls = renderer._wall_cache
    assert walls is not None
    alpha = pygame.surfarray.array_alpha(walls)
    grid = sim.level.grid
    floor_tiles = [
        (x, y)
        for y in range(grid.shape[0])
        for x in range(grid.shape[1])
        if grid[y, x] != Tile.WALL
    ]
    assert floor_tiles, "level has no floor tiles"
    for x, y in floor_tiles[:40]:
        centre = alpha[x * 32 + 16, y * 32 + 16]
        assert centre == 0, f"wall layer is opaque over floor tile {(x, y)}"


def test_cone_cache_matches_uncached_rays_while_guards_move() -> None:
    """Pose quantisation may shift a cone edge, but only imperceptibly."""

    def run(cached: bool) -> list[np.ndarray]:
        sim, renderer = _renderer(99, 6)
        if not cached:
            renderer._cone_ray_cache = _NeverRetains()
        frames = []
        for step in range(24):
            sim.advance(Action(move=1 + step % 8), ticks=6)
            frames.append(_frame(renderer))
        return frames

    worst = 0.0
    for warm, cold in zip(run(True), run(False)):
        differing = float((np.abs(warm.astype(np.int32) - cold.astype(np.int32)).max(axis=2) > 0).mean())
        worst = max(worst, differing)
    # Measured at 0.013% of pixels on the worst frame of a 270-frame sweep.
    assert worst < 0.005, f"cone cache changed {worst:.4%} of pixels; quantisation is too coarse"


def test_cone_cache_is_bounded() -> None:
    from ghostline.presentation import CONE_RAY_CACHE_LIMIT

    sim, renderer = _renderer(99, 6)
    for step in range(120):
        sim.advance(Action(move=1 + step % 8), ticks=6)
        renderer.draw()
    assert len(renderer._cone_ray_cache) <= CONE_RAY_CACHE_LIMIT


def test_world_projection_matches_the_vector_expression() -> None:
    """The scalar fast path must agree with the original NumPy formula."""

    sim, renderer = _renderer(7, 4)
    renderer.draw()
    centre = np.asarray(renderer._world_center())
    for point in ((0.0, 0.0), (17.5, 923.25), (512.0, 288.0), tuple(sim.player)):
        expected_vector = np.asarray(point) - renderer.camera + centre
        expected = (int(round(expected_vector[0])), int(round(expected_vector[1])))
        assert renderer._world(point) == expected
