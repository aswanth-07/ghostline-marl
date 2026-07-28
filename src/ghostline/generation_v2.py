"""Facility layout variation for the multi-agent (Env-v2) track.

`generation.py` is hashed into the published `GhostlineEnv-v1` contract, so it is
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

from collections import deque

import numpy as np

from ghostline.config import TILE_SIZE
from ghostline.config_v2 import (
    HACK_DEVICES_PER_TIER,
    VENT_MIN_PAIR_DISTANCE_TILES,
    VENT_PAIRS_PER_TIER,
)
from ghostline.generation import LevelGenerator, flood_fill, world_to_tile
from ghostline.generation import tile_center
from ghostline.types import GeneratedLevel, Prop, Tile
from ghostline.types_v2 import HackableDevice, Vent

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
DECOR_KINDS = ("floor_marking", "wall_sign", "cable_run")

# Only these tiers create lockable security doors in ``simulation_v2``.
# Keeping the table here lets generation avoid presenting an interaction that
# cannot have an effect, without importing the simulation into content code.
SECURITY_DOORS_PER_TIER = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3}
MAX_RESHAPE_ATTEMPTS = 8


class FacilityLayoutV2(LevelGenerator):
    """Env-v2 generator with deterministic content-readiness validation."""

    def generate(self, *, seed: int, tier: int) -> GeneratedLevel:
        # ``LevelGenerator.generate`` calls ``self.validate``.  V2 validation
        # intentionally requires vents and devices, which the classic level
        # does not have yet, so build the frozen baseline through its own
        # generator before applying the V2 passes.
        base = LevelGenerator().generate(seed=seed, tier=tier)
        for attempt in range(MAX_RESHAPE_ATTEMPTS):
            rng = np.random.default_rng(
                int(np.random.SeedSequence([seed, tier, 0x5A17, attempt]).generate_state(1)[0])
            )
            candidate = self._reshape(base, rng)
            if self.validate(candidate):
                return candidate
        # Missing field systems are a contract failure, not a presentation
        # fallback.  Failing deterministically is safer than admitting a seed
        # whose mechanics silently differ from the training distribution.
        raise RuntimeError(
            f"could not generate a valid Env-v2 tier {tier} level for seed {seed} "
            f"after {MAX_RESHAPE_ATTEMPTS} deterministic attempts"
        )

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
        # Every visible object, blocking or not, participates in one placement
        # reservation contract.  Essential field systems claim their cells
        # before flavour decor so a cosmetic pass can never hide an interaction.
        reserved = self._occupied_tiles(level)
        level.vents = self._place_vents(level, rng, protected, reserved)
        level.hackable = self._place_hackables(level, rng, protected, reserved)
        self._add_decor(level, rng, reserved)
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

    def _open_tiles(
        self,
        level: GeneratedLevel,
        protected: set[tuple[int, int]],
        reserved: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        blocked = self._blocked_by_props(level)
        reachable = flood_fill(level.grid, world_to_tile(level.spawn), blocked)
        return [
            (x, y)
            for y in range(1, level.grid.shape[0] - 1)
            for x in range(1, level.grid.shape[1] - 1)
            if level.grid[y, x] == Tile.FLOOR
            and (x, y) in reachable
            and (x, y) not in blocked
            and (x, y) not in protected
            and (x, y) not in reserved
            and self._safe_interaction_tile(level, (x, y), blocked)
        ]

    @staticmethod
    def _occupied_tiles(level: GeneratedLevel) -> set[tuple[int, int]]:
        """All rendered prop cells, including nonblocking flavour."""

        return {
            (prop.tile_x + dx, prop.tile_y + dy)
            for prop in level.props
            for dx in range(prop.width)
            for dy in range(prop.height)
        }

    @staticmethod
    def _walkable_tile(
        level: GeneratedLevel,
        tile: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> bool:
        x, y = tile
        return (
            0 <= y < level.grid.shape[0]
            and 0 <= x < level.grid.shape[1]
            and tile not in blocked
            and level.grid[y, x] != Tile.WALL
        )

    @classmethod
    def _safe_interaction_tile(
        cls,
        level: GeneratedLevel,
        tile: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> bool:
        """Require a reachable floor cell with more than one escape edge."""

        if not cls._walkable_tile(level, tile, blocked):
            return False
        x, y = tile
        exits = sum(
            cls._walkable_tile(level, (x + dx, y + dy), blocked)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        return exits >= 2

    @classmethod
    def _distance_map(
        cls,
        level: GeneratedLevel,
        start: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> dict[tuple[int, int], int]:
        """Cardinal geodesic distances over the simulation collision grid."""

        distances = {start: 0}
        pending = deque([start])
        while pending:
            x, y = pending.popleft()
            distance = distances[(x, y)] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                tile = (x + dx, y + dy)
                if tile in distances or not cls._walkable_tile(level, tile, blocked):
                    continue
                distances[tile] = distance
                pending.append(tile)
        return distances

    @staticmethod
    def _room_id_for_tile(level: GeneratedLevel, tile: tuple[int, int]) -> int | None:
        x, y = tile
        for room in level.rooms:
            if room.x <= x < room.x + room.width and room.y <= y < room.y + room.height:
                return int(room.room_id)
        return None

    @staticmethod
    def _edge_is_redundant(level: GeneratedLevel, room_a: int, room_b: int) -> bool:
        visited = {room_a}
        pending = [room_a]
        while pending:
            room = pending.pop()
            for neighbour in level.adjacency[room]:
                if {room, neighbour} == {room_a, room_b}:
                    continue
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        return room_b in visited

    @classmethod
    def _lockable_door_indices(cls, level: GeneratedLevel) -> list[int]:
        requested = SECURITY_DOORS_PER_TIER.get(level.tier, 0)
        if requested <= 0:
            return []
        candidates = [
            index
            for index, door in enumerate(level.doors)
            if cls._edge_is_redundant(level, door.room_a, door.room_b)
        ]
        candidates.sort(
            key=lambda index: (
                level.doors[index].tile[1],
                level.doors[index].tile[0],
                level.doors[index].room_a,
                level.doors[index].room_b,
            )
        )
        if candidates:
            offset = int(
                np.random.SeedSequence([level.seed, level.tier, 3001]).generate_state(1)[0]
            ) % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
        return candidates[:requested]

    def _place_vents(
        self,
        level: GeneratedLevel,
        rng: np.random.Generator,
        protected: set[tuple[int, int]],
        reserved: set[tuple[int, int]],
    ) -> list[Vent]:
        """Pair up maintenance ducts across distant rooms.

        Pairs are deliberately long-range: a vent is worth using because it
        crosses the facility, which is what makes it an answer to a sealed
        route rather than a shortcut around a corner.
        """

        wanted = VENT_PAIRS_PER_TIER.get(level.tier, 2)
        if wanted <= 0:
            return []
        blocked = self._blocked_by_props(level)
        candidates = [
            tile
            for tile in self._open_tiles(level, protected, reserved)
            if self._room_id_for_tile(level, tile) is not None
        ]
        if len(candidates) < 4:
            return []
        rng.shuffle(candidates)
        rank = {tile: index for index, tile in enumerate(candidates)}
        vents: list[Vent] = []
        used: set[tuple[int, int]] = set()
        for entry in candidates:
            if len(vents) >= wanted * 2:
                break
            if entry in used:
                continue
            entry_room = self._room_id_for_tile(level, entry)
            distances = self._distance_map(level, entry, blocked)
            partners = [
                other
                for other in candidates
                if other not in used
                and other != entry
                and self._room_id_for_tile(level, other) != entry_room
                and distances.get(other, -1) >= VENT_MIN_PAIR_DISTANCE_TILES
            ]
            if not partners:
                continue
            # Prefer the largest real route skip. Candidate shuffle order is a
            # deterministic tie-breaker, so the same seed always gets the same
            # network without settling for the first merely-valid endpoint.
            partner = max(partners, key=lambda other: (distances[other], -rank[other]))
            used.update({entry, partner})
            vents.append(Vent(len(vents), entry, partner, tile_center(partner)))
            vents.append(Vent(len(vents), partner, entry, tile_center(entry)))
            level.props.append(Prop("vent_shaft", entry[0], entry[1], 1, 1, False))
            level.props.append(Prop("vent_shaft", partner[0], partner[1], 1, 1, False))
            reserved.update({entry, partner})
        return vents

    def _place_hackables(
        self,
        level: GeneratedLevel,
        rng: np.random.Generator,
        protected: set[tuple[int, int]],
        reserved: set[tuple[int, int]],
    ) -> list[HackableDevice]:
        """Wall panels wired to a camera, a door, or the room lights."""

        wanted = HACK_DEVICES_PER_TIER.get(level.tier, 3)
        if wanted <= 0:
            return []
        candidates = self._open_tiles(level, protected, reserved)
        rng.shuffle(candidates)
        cameras_by_id = {int(camera.camera_id): camera for camera in level.cameras}
        camera_ids = list(cameras_by_id)
        lockable_indices = self._lockable_door_indices(level)
        used_targets: set[tuple[str, int]] = set()
        devices: list[HackableDevice] = []
        for tile in candidates:
            if len(devices) >= wanted:
                break
            room_id = self._room_id_for_tile(level, tile)
            weighted_kinds: list[str] = []
            if camera_ids:
                weighted_kinds.extend(("camera",) * 4)
            if lockable_indices:
                weighted_kinds.extend(("door",) * 3)
            if room_id is not None:
                weighted_kinds.extend(("lights",) * 3)
            if not weighted_kinds:
                continue
            kind = weighted_kinds[int(rng.integers(0, len(weighted_kinds)))]
            target_tile: tuple[int, int] | None = None
            if kind == "camera":
                ordered = sorted(
                    camera_ids,
                    key=lambda camera_id: (
                        (world_to_tile(cameras_by_id[camera_id].position)[0] - tile[0]) ** 2
                        + (world_to_tile(cameras_by_id[camera_id].position)[1] - tile[1]) ** 2,
                        camera_id,
                    ),
                )
                target = next(
                    (camera_id for camera_id in ordered if ("camera", camera_id) not in used_targets),
                    ordered[0],
                )
            elif kind == "door":
                ordered = sorted(
                    lockable_indices,
                    key=lambda door_index: (
                        (level.doors[door_index].tile[0] - tile[0]) ** 2
                        + (level.doors[door_index].tile[1] - tile[1]) ** 2,
                        door_index,
                    ),
                )
                target = next(
                    (door_index for door_index in ordered if ("door", door_index) not in used_targets),
                    ordered[0],
                )
                target_tile = level.doors[target].tile
            else:
                target = int(room_id)
            used_targets.add((kind, target))
            devices.append(
                HackableDevice(
                    len(devices),
                    kind,
                    tile,
                    tile_center(tile),
                    target,
                    target_tile=target_tile,
                )
            )
            level.props.append(Prop("hack_panel", tile[0], tile[1], 1, 1, False))
            reserved.add(tile)
        return devices

    def _add_decor(
        self,
        level: GeneratedLevel,
        rng: np.random.Generator,
        reserved: set[tuple[int, int]],
    ) -> None:
        """Add non-blocking flavour: floor markings, signage, and cabling.

        Decor never blocks movement or sight, so it cannot affect navigation,
        detection or any policy observation. It exists purely so a room reads as
        a place with a purpose rather than a rectangle with furniture in it.
        """

        budget = DECOR_BUDGET.get(level.tier, 12)
        for _ in range(budget):
            room = level.rooms[int(rng.integers(0, len(level.rooms)))]
            x = int(rng.integers(room.x + 1, max(room.x + 2, room.x + room.width - 1)))
            y = int(rng.integers(room.y + 1, max(room.y + 2, room.y + room.height - 1)))
            if not (0 <= y < level.grid.shape[0] and 0 <= x < level.grid.shape[1]):
                continue
            if level.grid[y, x] != Tile.FLOOR or (x, y) in reserved:
                continue
            kind = DECOR_KINDS[int(rng.integers(0, len(DECOR_KINDS)))]
            level.props.append(Prop(kind, x, y, 1, 1, False))
            reserved.add((x, y))

    # -- V2 contract validation --------------------------------------------

    def validate(self, level: GeneratedLevel) -> bool:
        """Validate classic guarantees plus every authored V2 field system."""

        return not self.readiness_errors(level)

    def readiness_errors(self, level: GeneratedLevel) -> tuple[str, ...]:
        """Return deterministic, machine-readable V2 contract failures."""

        errors: list[str] = []
        if not super().validate(level):
            errors.append("classic_contract")
            return tuple(errors)

        vents = getattr(level, "vents", None)
        devices = getattr(level, "hackable", None)
        if vents is None:
            errors.append("missing_vents")
            vents = []
        if devices is None:
            errors.append("missing_hackable")
            devices = []

        expected_vents = VENT_PAIRS_PER_TIER.get(level.tier, 0) * 2
        expected_devices = HACK_DEVICES_PER_TIER.get(level.tier, 0)
        if len(vents) != expected_vents:
            errors.append("vent_count")
        if len(devices) != expected_devices:
            errors.append("device_count")

        # A prop tile has exactly one visual owner. This catches not only a
        # vent hidden by a panel, but also cosmetic decals obscuring either.
        occupied: dict[tuple[int, int], str] = {}
        for prop in level.props:
            for dx in range(prop.width):
                for dy in range(prop.height):
                    tile = (prop.tile_x + dx, prop.tile_y + dy)
                    if tile in occupied:
                        errors.append("prop_overlap")
                    else:
                        occupied[tile] = prop.kind

        blocked = self._blocked_by_props(level)
        reachable = flood_fill(level.grid, world_to_tile(level.spawn), blocked)
        vent_tiles = [vent.tile for vent in vents]
        device_tiles = [device.tile for device in devices]
        if len(set(vent_tiles)) != len(vent_tiles):
            errors.append("duplicate_vent_tile")
        if len(set(device_tiles)) != len(device_tiles):
            errors.append("duplicate_device_tile")
        if set(vent_tiles) & set(device_tiles):
            errors.append("field_overlap")
        if len({vent.vent_id for vent in vents}) != len(vents):
            errors.append("duplicate_vent_id")
        if len({device.device_id for device in devices}) != len(devices):
            errors.append("duplicate_device_id")

        vent_props = {
            (prop.tile_x, prop.tile_y)
            for prop in level.props
            if prop.kind == "vent_shaft" and prop.width == 1 and prop.height == 1
        }
        panel_props = {
            (prop.tile_x, prop.tile_y)
            for prop in level.props
            if prop.kind == "hack_panel" and prop.width == 1 and prop.height == 1
        }
        if vent_props != set(vent_tiles):
            errors.append("vent_prop_binding")
        if panel_props != set(device_tiles):
            errors.append("panel_prop_binding")

        active_tiles = [*vent_tiles, *device_tiles]
        if any(tile not in reachable for tile in active_tiles):
            errors.append("unreachable_interaction")
        if any(not self._safe_interaction_tile(level, tile, blocked) for tile in active_tiles):
            errors.append("unsafe_interaction")

        by_tile = {vent.tile: vent for vent in vents}
        distance_cache: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        for vent in vents:
            partner = by_tile.get(vent.exit_tile)
            if partner is None or partner.exit_tile != vent.tile:
                errors.append("unpaired_vent")
                continue
            if not np.allclose(vent.exit_position, tile_center(vent.exit_tile), atol=1e-5):
                errors.append("vent_exit_position")
            room = self._room_id_for_tile(level, vent.tile)
            exit_room = self._room_id_for_tile(level, vent.exit_tile)
            if room is None or exit_room is None or room == exit_room:
                errors.append("vent_room_pair")
            distances = distance_cache.setdefault(
                vent.tile,
                self._distance_map(level, vent.tile, blocked),
            )
            if distances.get(vent.exit_tile, -1) < VENT_MIN_PAIR_DISTANCE_TILES:
                errors.append("vent_geodesic")

        camera_ids = {int(camera.camera_id) for camera in level.cameras}
        room_ids = {int(room.room_id) for room in level.rooms}
        lockable_indices = set(self._lockable_door_indices(level))
        for device in devices:
            if device.kind == "camera":
                if device.target_id not in camera_ids:
                    errors.append("camera_target")
                if device.target_tile is not None:
                    errors.append("camera_target_tile")
            elif device.kind == "door":
                if SECURITY_DOORS_PER_TIER.get(level.tier, 0) <= 0:
                    errors.append("useless_door_panel")
                if device.target_id not in lockable_indices:
                    errors.append("door_target")
                elif device.target_tile != level.doors[device.target_id].tile:
                    errors.append("door_target_tile")
            elif device.kind == "lights":
                if device.target_id not in room_ids:
                    errors.append("lights_target")
                if self._room_id_for_tile(level, device.tile) != device.target_id:
                    errors.append("lights_panel_room")
                if device.target_tile is not None:
                    errors.append("lights_target_tile")
            else:
                errors.append("unknown_device_kind")

        # Keep output stable and compact for fuzz tooling.
        return tuple(dict.fromkeys(errors))


def generate_v2_level(*, seed: int, tier: int) -> GeneratedLevel:
    """Convenience entry point mirroring ``LevelGenerator.generate``."""

    return FacilityLayoutV2().generate(seed=seed, tier=tier)
