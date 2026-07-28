"""Facility layout variation for the multi-agent (Env-v3) track.

`generation.py` is hashed into the frozen `GhostlineEnv-v2` contract, so it is
never touched here. This module takes a fully generated, already valid level and
reshapes its geometry: uniform 11x9 boxes become rooms with alcoves, recessed
corners, interior partitions and pillars, and corridors gain junctions and
niches instead of running dead straight.

The approach is deliberately subtractive and conservative. Every candidate wall
is checked against the objects that must stay reachable and against a flood fill
before it is committed, and the finished level is run through the original
`LevelGenerator.validate`. That reuses all of the proven placement, security
buffer and route-redundancy machinery instead of reimplementing it, so a layout
change cannot silently break reachability, door throats or camera clearances.

Design intent, in priority order:

1. Break the grid read. A player should not be able to predict a room's shape
   from its position, which is what made the original facilities feel samey.
2. Create real cover and sightline structure. Partitions and pillars produce
   short sightlines and corner peeking, which is what the crouch/cover stealth
   layer and the security interception shaping both need to be interesting.
3. Preserve every guarantee. Reachability, route redundancy and the security
   clearances are contracts, not preferences.
"""

from __future__ import annotations

import numpy as np

from ghostline.config import TILE_SIZE
from ghostline.generation import LevelGenerator, flood_fill, world_to_tile
from ghostline.types import GeneratedLevel, Prop, Tile

# Per-tier carving budget. Later tiers are larger, so they can absorb more
# structure before the facility starts to feel like a maze.
CARVE_BUDGET = {1: 10, 2: 16, 3: 26, 4: 34, 5: 44, 6: 56}
# Interior structure that blocks sight and movement but reads as furniture.
PILLAR_BUDGET = {1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9}
# Structured interior runs -- rack aisles, desk rows, shelving. This is what
# actually breaks the "big empty rectangle" read; corner carving alone only
# changes a silhouette the player rarely sees all of at once.
AISLE_ROOM_BUDGET = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 8}
# Per role: the prop laid down in runs, and whether runs are vertical.
ROOM_ARCHETYPES = {
    "server": ("server", True),
    "vault": ("vault_case", False),
    "utility": ("crate", True),
    "office": ("desk", False),
    "lab": ("lab_bench", False),
    "lounge": ("sofa", False),
    "security": ("locker", True),
}
# Free-standing decor that never blocks: floor markings, signage, cabling.
DECOR_BUDGET = {1: 6, 2: 9, 3: 14, 4: 18, 5: 22, 6: 28}
DECOR_KINDS = ("floor_marking", "wall_sign", "cable_run", "vent_grate")


class FacilityLayoutV3(LevelGenerator):
    """Env-v3 generator: the classic facility, reshaped for character."""

    def generate(self, *, seed: int, tier: int) -> GeneratedLevel:
        base = super().generate(seed=seed, tier=tier)
        for attempt in range(8):
            rng = np.random.default_rng(
                int(np.random.SeedSequence([seed, tier, 0x5A17, attempt]).generate_state(1)[0])
            )
            candidate = self._reshape(base, rng)
            if self.validate(candidate):
                return candidate
        # Variation is a presentation and tactics improvement, never a
        # correctness requirement: an unusually tight seed keeps its original
        # rectangular layout rather than failing to produce a level at all.
        return base

    # -- geometry -----------------------------------------------------------

    def _protected_tiles(self, level: GeneratedLevel) -> set[tuple[int, int]]:
        """Tiles that must stay walkable for the level to remain valid."""

        protected: set[tuple[int, int]] = {
            world_to_tile(level.spawn),
            world_to_tile(level.extraction),
        }
        protected.update(world_to_tile(terminal.position) for terminal in level.terminals)
        protected.update(world_to_tile(camera.position) for camera in level.cameras)
        for guard in level.guards:
            protected.add(world_to_tile(guard.position))
            protected.update(world_to_tile(point) for point in guard.patrol)
        for door in level.doors:
            x, y = door.tile
            # Door throats need their approaches on both sides.
            protected.update({(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)})
        for prop in level.props:
            for dx in range(prop.width):
                for dy in range(prop.height):
                    protected.add((prop.tile_x + dx, prop.tile_y + dy))
        # Keep a one-tile ring around every objective so link pockets survive.
        ringed = set(protected)
        for x, y in protected:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ringed.add((x + dx, y + dy))
        return ringed

    def _blocked_by_props(self, level: GeneratedLevel) -> set[tuple[int, int]]:
        return {
            (prop.tile_x + dx, prop.tile_y + dy)
            for prop in level.props
            if prop.blocking
            for dx in range(prop.width)
            for dy in range(prop.height)
        }

    def _reshape(self, base: GeneratedLevel, rng: np.random.Generator) -> GeneratedLevel:
        grid = base.grid.copy()
        props = [Prop(p.kind, p.tile_x, p.tile_y, p.width, p.height, p.blocking) for p in base.props]
        level = GeneratedLevel(
            seed=base.seed,
            tier=base.tier,
            grid=grid,
            rooms=base.rooms,
            doors=base.doors,
            props=props,
            terminals=base.terminals,
            cameras=base.cameras,
            guards=base.guards,
            spawn=base.spawn,
            extraction=base.extraction,
            quota=base.quota,
            mission_seconds=base.mission_seconds,
            pulse_charges=base.pulse_charges,
            response_drones=base.response_drones,
            drone_trace_threshold=base.drone_trace_threshold,
            adjacency=base.adjacency,
        )
        protected = self._protected_tiles(level)
        blocked = self._blocked_by_props(level)
        start = world_to_tile(level.spawn)
        required = [world_to_tile(level.extraction), *(world_to_tile(t.position) for t in level.terminals)]

        self._carve_room_character(level, rng, protected, blocked, start, required)
        self._add_room_archetypes(level, rng, protected, blocked, start, required)
        self._add_interior_structure(level, rng, protected, blocked, start, required)
        self._add_decor(level, rng)
        return level

    @staticmethod
    def _connectivity_ok(
        level: GeneratedLevel,
        blocked: set[tuple[int, int]],
        start: tuple[int, int],
        required: list[tuple[int, int]],
    ) -> bool:
        reachable = flood_fill(level.grid, start, blocked)
        walkable = int(np.count_nonzero(level.grid != Tile.WALL))
        return all(item in reachable for item in required) and len(reachable) >= max(
            12, int(walkable * 0.72)
        )

    def _eligible(
        self,
        level: GeneratedLevel,
        tile: tuple[int, int],
        protected: set[tuple[int, int]],
    ) -> bool:
        x, y = tile
        if tile in protected:
            return False
        if not (0 <= y < level.grid.shape[0] and 0 <= x < level.grid.shape[1]):
            return False
        return bool(level.grid[y, x] == Tile.FLOOR)

    def _commit_walls(
        self,
        level: GeneratedLevel,
        tiles: list[tuple[int, int]],
        protected: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
        start: tuple[int, int],
        required: list[tuple[int, int]],
    ) -> int:
        """Apply a batch of walls behind a single connectivity check.

        Flood filling once per candidate tile cost roughly a 10x slowdown
        against the base generator, which matters because levels are generated
        on every training reset. The batch is applied optimistically and only
        unwound tile by tile on the rare occasion it breaks reachability.
        """

        applied = [tile for tile in tiles if self._eligible(level, tile, protected)]
        if not applied:
            return 0
        for x, y in applied:
            level.grid[y, x] = Tile.WALL
        if self._connectivity_ok(level, blocked, start, required):
            return len(applied)
        # Unwind newest-first until the facility reconnects.
        for x, y in reversed(applied):
            level.grid[y, x] = Tile.FLOOR
            if self._connectivity_ok(level, blocked, start, required):
                kept = applied[: applied.index((x, y))]
                return len(kept)
        return 0

    def _carve_room_character(
        self,
        level: GeneratedLevel,
        rng: np.random.Generator,
        protected: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
        start: tuple[int, int],
        required: list[tuple[int, int]],
    ) -> None:
        """Recess corners and edges so rooms stop reading as identical boxes.

        Candidates for the whole facility are gathered first and committed in a
        single batch. Checking connectivity per room cost about 25 flood fills
        per level, which dominated generation time on a path that runs at every
        training reset.
        """

        budget = CARVE_BUDGET.get(level.tier, 24)
        order = list(level.rooms)
        rng.shuffle(order)
        batch: list[tuple[int, int]] = []
        for room in order:
            if len(batch) >= budget:
                break
            # Each room picks its own silhouette treatment, so neighbours differ.
            style = int(rng.integers(0, 4))
            left, top = room.x + 1, room.y + 1
            right, bottom = room.x + room.width - 2, room.y + room.height - 2
            if right - left < 4 or bottom - top < 4:
                continue
            depth = int(rng.integers(1, 3))
            if style == 0:  # single recessed corner -> L-shaped room
                corners = [(left, top)]
            elif style == 1:  # opposite corners -> stepped room
                corners = [(left, top), (right, bottom)]
            elif style == 2:  # one full side alcove
                corners = [(left, top), (left, bottom)]
            else:  # narrow the room waist
                corners = [(right, top)]
            for corner_x, corner_y in corners:
                step_x = 1 if corner_x == left else -1
                step_y = 1 if corner_y == top else -1
                for dx in range(depth):
                    for dy in range(depth):
                        batch.append((corner_x + dx * step_x, corner_y + dy * step_y))
        self._commit_walls(level, batch[:budget], protected, blocked, start, required)

    def _add_room_archetypes(
        self,
        level: GeneratedLevel,
        rng: np.random.Generator,
        protected: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
        start: tuple[int, int],
        required: list[tuple[int, int]],
    ) -> None:
        """Lay structured runs so a room reads as a place with a function.

        A server room becomes rack aisles, an office becomes desk rows, storage
        becomes shelving. Runs leave a walkable gap between them, which is the
        point for gameplay as well as looks: parallel aisles create short
        sightlines, corner peeking and real cover, which is exactly the geometry
        the crouch stealth layer and the security interception shaping need.
        """

        budget = AISLE_ROOM_BUDGET.get(level.tier, 4)
        rooms = [room for room in level.rooms if room.role in ROOM_ARCHETYPES]
        rng.shuffle(rooms)
        placed_rooms = 0
        additions: list[tuple[str, tuple[int, int]]] = []
        taken: set[tuple[int, int]] = set()
        for room in rooms:
            if placed_rooms >= budget:
                break
            kind, vertical = ROOM_ARCHETYPES[room.role]
            left, top = room.x + 2, room.y + 2
            right, bottom = room.x + room.width - 3, room.y + room.height - 3
            if right - left < 4 or bottom - top < 4:
                continue
            # Runs every third line leaves a clear walkway between them.
            span = range(left, right + 1) if vertical else range(top, bottom + 1)
            lines = [value for index, value in enumerate(span) if index % 3 == 1]
            if not lines:
                continue
            room_additions: list[tuple[str, tuple[int, int]]] = []
            for line in lines:
                cross = range(top, bottom + 1) if vertical else range(left, right + 1)
                # Leave the ends open so a run never seals a corner.
                inner = list(cross)[1:-1]
                if len(inner) < 2:
                    continue
                for value in inner:
                    tile = (line, value) if vertical else (value, line)
                    x, y = tile
                    if level.grid[y, x] != Tile.FLOOR:
                        continue
                    if tile in protected or tile in blocked or tile in taken:
                        continue
                    room_additions.append((kind, tile))
                    taken.add(tile)
            if room_additions:
                additions.extend(room_additions)
                placed_rooms += 1
        if not additions:
            return
        trial = blocked | {tile for _, tile in additions}
        reachable = flood_fill(level.grid, start, trial)
        if not all(item in reachable for item in required):
            # Rather than lose all interior structure, retry with alternating
            # runs only, which always leaves a wider walkway.
            additions = additions[::2]
            trial = blocked | {tile for _, tile in additions}
            reachable = flood_fill(level.grid, start, trial)
            if not all(item in reachable for item in required):
                return
        for kind, tile in additions:
            level.props.append(Prop(kind, tile[0], tile[1], 1, 1, True))
            blocked.add(tile)
            protected.add(tile)

    def _add_interior_structure(
        self,
        level: GeneratedLevel,
        rng: np.random.Generator,
        protected: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
        start: tuple[int, int],
        required: list[tuple[int, int]],
    ) -> None:
        """Drop pillars and short partitions that break long sightlines.

        These are props rather than wall tiles so they render as facility
        structure and keep the room's floor material, and because the simulation
        already treats blocking props as occluders for sight and navigation.
        Like the carve pass, the whole batch shares one connectivity check.
        """

        budget = PILLAR_BUDGET.get(level.tier, 4)
        rooms = list(level.rooms)
        rng.shuffle(rooms)
        chosen: list[tuple[str, tuple[int, int]]] = []
        taken: set[tuple[int, int]] = set()
        for room in rooms:
            if len(chosen) >= budget:
                break
            interior = [
                (x, y)
                for y in range(room.y + 2, room.y + room.height - 2)
                for x in range(room.x + 2, room.x + room.width - 2)
                if level.grid[y, x] == Tile.FLOOR
                and (x, y) not in protected
                and (x, y) not in blocked
                and (x, y) not in taken
            ]
            if len(interior) < 6:
                continue
            tile = interior[int(rng.integers(0, len(interior)))]
            kind = "pillar" if rng.random() < 0.55 else "partition"
            chosen.append((kind, tile))
            taken.add(tile)
        if not chosen:
            return
        trial = blocked | {tile for _, tile in chosen}
        reachable = flood_fill(level.grid, start, trial)
        if not all(item in reachable for item in required):
            # Fall back to the single safest pillar rather than dropping the
            # whole pass: interior structure is what makes rooms readable.
            chosen = chosen[:1]
            trial = blocked | {chosen[0][1]}
            reachable = flood_fill(level.grid, start, trial)
            if not all(item in reachable for item in required):
                return
        for kind, tile in chosen:
            level.props.append(Prop(kind, tile[0], tile[1], 1, 1, True))
            blocked.add(tile)
            protected.add(tile)

    def _add_decor(self, level: GeneratedLevel, rng: np.random.Generator) -> None:
        """Add non-blocking flavour: floor markings, signage, cabling, vents.

        Decor never blocks movement or sight, so it cannot affect navigation,
        detection or any policy observation. It exists purely so a room reads as
        a place with a purpose rather than a rectangle with furniture in it.
        """

        budget = DECOR_BUDGET.get(level.tier, 12)
        occupied = {
            (prop.tile_x + dx, prop.tile_y + dy)
            for prop in level.props
            for dx in range(prop.width)
            for dy in range(prop.height)
        }
        for _ in range(budget):
            room = level.rooms[int(rng.integers(0, len(level.rooms)))]
            x = int(rng.integers(room.x + 1, max(room.x + 2, room.x + room.width - 1)))
            y = int(rng.integers(room.y + 1, max(room.y + 2, room.y + room.height - 1)))
            if not (0 <= y < level.grid.shape[0] and 0 <= x < level.grid.shape[1]):
                continue
            if level.grid[y, x] != Tile.FLOOR or (x, y) in occupied:
                continue
            kind = DECOR_KINDS[int(rng.integers(0, len(DECOR_KINDS)))]
            level.props.append(Prop(kind, x, y, 1, 1, False))
            occupied.add((x, y))


def generate_v3_level(*, seed: int, tier: int) -> GeneratedLevel:
    """Convenience entry point mirroring ``LevelGenerator.generate``."""

    return FacilityLayoutV3().generate(seed=seed, tier=tier)
