from __future__ import annotations

from collections import deque
from dataclasses import replace
import math

import numpy as np

from ghostline.config import (
    CAMERA_SWEEP_RADIANS,
    CAMERA_VISION_COSINE,
    CAMERA_VISION_DISTANCE,
    HACK_RADIUS,
    ROOM_HEIGHT_TILES,
    ROOM_WIDTH_TILES,
    TILE_SIZE,
    TIERS,
)
from ghostline.types import Camera, Door, GeneratedLevel, Guard, GuardGrade, Prop, Room, Terminal, Tile

ROOM_ROLES = ("office", "lounge", "lab", "server", "security", "vault", "utility", "corridor")

# These fixtures are authored as banks of identical modules.  Keeping their
# collision as one long Prop made the renderer bottom-pivot a single atlas crop
# across the whole footprint: a 1x4 server bank consequently drew art only in
# its final tile while the first three tiles still blocked movement.  Expanding
# the bank into deterministic one-tile modules preserves the exact occupied
# tile set and simulation rules while giving every collision cell its own
# visible rack/locker/seat/console.
MODULAR_PROP_KINDS = frozenset(("server", "locker"))

CAMERA_HALF_ANGLE = math.acos(CAMERA_VISION_COSINE)
CAMERA_OBJECTIVE_CLEARANCE = 5
GUARD_OBJECTIVE_CLEARANCE = 4
DOOR_SECURITY_CLEARANCE = 2

# A little redundancy in the room graph is an important stealth affordance:
# when a patrol or camera owns one doorway, higher-tier contracts must still
# offer another route instead of becoming a forced detection check.
MIN_GRAPH_LOOPS = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}

# Authored furniture arrangements. Coordinates are local interior tiles.
PROP_VARIANTS: dict[str, tuple[tuple[tuple[str, int, int, int, int], ...], ...]] = {
    "office": (
        (("desk", 2, 2, 2, 1), ("chair", 2, 3, 1, 1), ("desk", 6, 5, 2, 1), ("plant", 8, 1, 1, 1)),
        (("meeting_table", 3, 3, 4, 2), ("tv", 8, 3, 1, 2), ("plant", 1, 6, 1, 1)),
        (("desk", 2, 5, 2, 1), ("chair", 3, 4, 1, 1), ("desk", 7, 2, 2, 1), ("plant", 8, 6, 1, 1)),
    ),
    "lounge": (
        (("sofa", 2, 2, 3, 1), ("coffee_table", 3, 4, 2, 1), ("tv", 8, 2, 1, 2), ("plant", 1, 6, 1, 1)),
        (("sofa", 2, 5, 3, 1), ("sofa", 6, 2, 1, 3), ("coffee_table", 4, 3, 2, 2)),
        (("sofa", 1, 3, 1, 3), ("coffee_table", 3, 3, 2, 2), ("tv", 8, 4, 1, 2), ("plant", 7, 1, 1, 1)),
    ),
    "lab": (
        (("lab_bench", 2, 2, 2, 2), ("lab_bench", 6, 4, 2, 2), ("monitor", 7, 2, 1, 1)),
        (("lab_bench", 2, 5, 3, 1), ("cabinet", 7, 2, 1, 3), ("monitor", 3, 2, 1, 1)),
        (("cabinet", 1, 2, 1, 3), ("lab_bench", 3, 2, 3, 1), ("lab_bench", 6, 5, 3, 1), ("monitor", 8, 2, 1, 1)),
    ),
    "server": (
        (("server", 2, 2, 1, 4), ("server", 5, 2, 1, 4), ("server", 8, 2, 1, 4)),
        (("server", 2, 2, 3, 1), ("server", 6, 2, 3, 1), ("server", 2, 5, 3, 1), ("server", 6, 5, 3, 1)),
        (("server", 1, 2, 1, 4), ("server", 4, 1, 3, 1), ("server", 8, 2, 1, 4), ("console", 4, 6, 3, 1)),
    ),
    "security": (
        (("console", 2, 2, 3, 1), ("console", 6, 2, 3, 1), ("locker", 8, 5, 1, 2)),
        (("console", 4, 3, 3, 2), ("locker", 1, 2, 1, 3), ("locker", 8, 2, 1, 3)),
        (("locker", 2, 1, 3, 1), ("console", 2, 5, 3, 1), ("console", 7, 3, 2, 2)),
    ),
    "vault": (
        (("vault_case", 3, 2, 4, 2), ("server", 2, 5, 1, 2), ("server", 8, 5, 1, 2)),
        (("vault_case", 3, 4, 4, 2), ("console", 3, 2, 4, 1)),
        (("vault_case", 2, 2, 2, 2), ("vault_case", 7, 4, 2, 2), ("console", 4, 6, 3, 1)),
    ),
    "utility": (
        (("crate", 2, 2, 2, 2), ("generator", 7, 4, 2, 2)),
        (("generator", 2, 4, 2, 2), ("crate", 6, 2, 2, 2), ("locker", 8, 5, 1, 2)),
        (("locker", 1, 2, 1, 3), ("generator", 4, 2, 2, 2), ("crate", 7, 5, 2, 2)),
    ),
    "corridor": ((),),
    "extraction": ((('console', 2, 2, 2, 1),),),
}


def tile_center(tile: tuple[int, int]) -> np.ndarray:
    return np.asarray(((tile[0] + 0.5) * TILE_SIZE, (tile[1] + 0.5) * TILE_SIZE), dtype=np.float32)


class LevelGenerator:
    """Deterministic modular facility generator with validation and retry."""

    def generate(self, *, seed: int, tier: int) -> GeneratedLevel:
        if tier not in TIERS:
            raise ValueError(f"tier must be 1..6, got {tier}")
        for attempt in range(32):
            attempt_seed = int(np.random.SeedSequence([seed, tier, attempt]).generate_state(1)[0])
            level = self._build(seed=seed, tier=tier, rng=np.random.default_rng(attempt_seed))
            if self.validate(level):
                return level
        raise RuntimeError(f"could not generate a valid tier {tier} level for seed {seed}")

    def _build(self, *, seed: int, tier: int, rng: np.random.Generator) -> GeneratedLevel:
        spec = TIERS[tier]
        cols, rows = spec.room_columns, spec.room_rows
        width = cols * ROOM_WIDTH_TILES + 1
        height = rows * ROOM_HEIGHT_TILES + 1
        grid = np.full((height, width), Tile.WALL, dtype=np.int8)
        rooms: list[Room] = []

        # Deal role cards without replacement before repeating the deck. This
        # makes even the compact Orientation/Surveillance layouts visually and
        # tactically varied while keeping the result entirely seed-driven.
        role_deck = list(ROOM_ROLES)
        rng.shuffle(role_deck)
        interior_index = 0
        for row in range(rows):
            for col in range(cols):
                room_id = row * cols + col
                if room_id == 0:
                    role = "office"
                elif room_id == cols * rows - 1:
                    role = "extraction"
                else:
                    if interior_index and interior_index % len(role_deck) == 0:
                        rng.shuffle(role_deck)
                    role = role_deck[interior_index % len(role_deck)]
                    interior_index += 1
                x, y = col * ROOM_WIDTH_TILES, row * ROOM_HEIGHT_TILES
                variant = int(rng.integers(0, len(PROP_VARIANTS[role])))
                room = Room(room_id, col, row, role, x, y, ROOM_WIDTH_TILES + 1, ROOM_HEIGHT_TILES + 1, variant)
                rooms.append(room)
                grid[y + 1 : y + ROOM_HEIGHT_TILES, x + 1 : x + ROOM_WIDTH_TILES] = Tile.FLOOR

        candidates: list[tuple[int, int]] = []
        for room in rooms:
            if room.column + 1 < cols:
                candidates.append((room.room_id, room.room_id + 1))
            if room.row + 1 < rows:
                candidates.append((room.room_id, room.room_id + cols))
        rng.shuffle(candidates)

        parent = list(range(len(rooms)))

        def root(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        edges: list[tuple[int, int]] = []
        leftovers: list[tuple[int, int]] = []
        for a, b in candidates:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[rb] = ra
                edges.append((a, b))
            else:
                leftovers.append((a, b))
        selected_loops = 0
        deferred_loops: list[tuple[int, int]] = []
        for edge in leftovers:
            if rng.random() < spec.loop_chance:
                edges.append(edge)
                selected_loops += 1
            else:
                deferred_loops.append(edge)
        required_loops = min(MIN_GRAPH_LOOPS[tier], len(leftovers))
        if selected_loops < required_loops:
            edges.extend(deferred_loops[: required_loops - selected_loops])

        doors: list[Door] = []
        adjacency = {room.room_id: [] for room in rooms}
        for a, b in edges:
            room_a, room_b = rooms[a], rooms[b]
            if room_a.row == room_b.row:
                x = max(room_a.x, room_b.x)
                y = room_a.y + ROOM_HEIGHT_TILES // 2
            else:
                x = room_a.x + ROOM_WIDTH_TILES // 2
                y = max(room_a.y, room_b.y)
            grid[y, x] = Tile.DOOR
            doors.append(Door((x, y), a, b))
            adjacency[a].append(b)
            adjacency[b].append(a)

        props = self._place_props(rooms, grid, rng)
        blocked = {(p.tile_x + dx, p.tile_y + dy) for p in props if p.blocking for dx in range(p.width) for dy in range(p.height)}

        spawn_tile = self._clear_tile(rooms[0].center_tile, grid, blocked)
        extraction_tile = self._clear_tile(rooms[-1].center_tile, grid, blocked)
        preferred_terminal_rooms = [room for room in rooms[1:-1] if room.role != "corridor"]
        fallback_terminal_rooms = [room for room in rooms[1:-1] if room.role == "corridor"]
        rng.shuffle(preferred_terminal_rooms)
        rng.shuffle(fallback_terminal_rooms)
        eligible = preferred_terminal_rooms + fallback_terminal_rooms
        terminals: list[Terminal] = []
        for index in range(spec.terminal_count):
            room = eligible[index]
            terminal_tile = self._terminal_tile(room, grid, blocked, index=index)
            high_value_room = room.role in ("server", "security", "vault")
            value = 1 + int(high_value_room or rng.random() > 0.58)
            value += int(tier >= 5 and high_value_room and rng.random() > 0.62)
            terminals.append(Terminal(index, tile_center(terminal_tile), min(3, value), hack_seconds=0.0))

        # Quota is guaranteed by construction rather than by repeatedly hoping
        # random values add up. Promotions favour specialist rooms, so optional
        # vault/server detours communicate a meaningful risk/value choice.
        promotion_order = sorted(
            terminals,
            key=lambda terminal: (
                rooms[self.room_at_tile(rooms, world_to_tile(terminal.position))].role
                not in ("vault", "server", "security"),
                terminal.terminal_id,
            ),
        )
        while sum(terminal.value for terminal in terminals) < spec.quota:
            promotable = next((terminal for terminal in promotion_order if terminal.value < 3), None)
            if promotable is None:
                break
            promotable.value += 1
        for terminal in terminals:
            terminal.hack_seconds = 0.78 + 0.22 * terminal.value

        objective_tiles = {world_to_tile(terminal.position) for terminal in terminals} | {spawn_tile, extraction_tile}
        door_tiles = {door.tile for door in doors}
        camera_positions = self._sample_security_tiles(
            rooms,
            grid,
            blocked,
            spec.cameras,
            rng,
            avoid=objective_tiles,
            avoid_radius=CAMERA_OBJECTIVE_CLEARANCE,
            doors=door_tiles,
            prefer_perimeter=True,
        )
        cameras: list[Camera] = []
        for index, tile in enumerate(camera_positions):
            base_angle = self._camera_base_angle(
                tile,
                grid,
                blocked,
                objective_tiles,
                phase=float(rng.uniform(0.0, math.tau)),
            )
            cameras.append(Camera(index, tile_center(tile), angle=base_angle, base_angle=base_angle))
        guard_tiles = self._sample_security_tiles(
            rooms, grid, blocked | set(camera_positions), spec.guards, rng,
            avoid=objective_tiles,
            avoid_radius=GUARD_OBJECTIVE_CLEARANCE,
            doors=door_tiles,
        )
        guards: list[Guard] = []
        for index, tile in enumerate(guard_tiles):
            room_id = self.room_at_tile(rooms, tile)
            patrol_tiles = self._patrol_route(
                start=tile,
                room_id=room_id,
                rooms=rooms,
                doors=doors,
                adjacency=adjacency,
                grid=grid,
                blocked=blocked,
                route_index=index,
            )
            guards.append(
                Guard(
                    index,
                    tile_center(tile),
                    [tile_center(p) for p in patrol_tiles],
                    grade=self._guard_grade(tier, index, spec.guards),
                )
            )

        return GeneratedLevel(
            seed=seed,
            tier=tier,
            grid=grid,
            rooms=rooms,
            doors=doors,
            props=props,
            terminals=terminals,
            cameras=cameras,
            guards=guards,
            spawn=tile_center(spawn_tile),
            extraction=tile_center(extraction_tile),
            quota=spec.quota,
            mission_seconds=spec.mission_seconds,
            pulse_charges=spec.pulse_charges,
            response_drones=spec.response_drones,
            drone_trace_threshold=spec.drone_trace_threshold,
            adjacency=adjacency,
        )

    @staticmethod
    def _guard_grade(tier: int, index: int, count: int) -> GuardGrade:
        """Deal readable security grades as contract difficulty escalates."""

        if tier <= 3:
            return GuardGrade.STANDARD
        if tier == 4:
            return GuardGrade.INTERCEPTOR if index == count - 1 else GuardGrade.STANDARD
        if tier == 5:
            return (GuardGrade.STANDARD, GuardGrade.INTERCEPTOR, GuardGrade.ELITE)[index % 3]
        # Ghostline uses a readable five-operative threat pyramid: three
        # Standard, one Interceptor, and one 99%-speed Elite. Five cameras and
        # the trace-triggered drone retain full-system pressure without a six-
        # guard doorway stack.
        distribution = (
            GuardGrade.STANDARD,
            GuardGrade.STANDARD,
            GuardGrade.STANDARD,
            GuardGrade.INTERCEPTOR,
            GuardGrade.ELITE,
        )
        return distribution[min(len(distribution) - 1, index)]

    def _place_props(self, rooms: list[Room], grid: np.ndarray, rng: np.random.Generator) -> list[Prop]:
        props: list[Prop] = []
        door_buffer = {(door_x, door_y) for door_y, door_x in np.argwhere(grid == Tile.DOOR)}
        for room in rooms:
            if room.room_id in (0, len(rooms) - 1):
                continue
            layout = PROP_VARIANTS[room.role][room.variant]
            for kind, ox, oy, width, height in layout:
                occupied = {(room.x + ox + dx, room.y + oy + dy) for dx in range(width) for dy in range(height)}
                if any(abs(x - dx) + abs(y - dy) <= 1 for x, y in occupied for dx, dy in door_buffer):
                    continue
                props.extend(self._expand_visible_fixture(kind, room.x + ox, room.y + oy, width, height))
        return props

    @staticmethod
    def _expand_visible_fixture(kind: str, tile_x: int, tile_y: int, width: int, height: int) -> list[Prop]:
        """Return props whose rendered modules cover the same blocked cells.

        The environment atlas contains a single rack, locker, sofa, or console
        module rather than arbitrary stretched banks.  Split only footprints
        whose single bottom-pivot crop leaves whole (or nearly whole) collision
        cells visually empty.  The row-major order is stable and the union of
        occupied tiles is exactly unchanged.
        """

        split_into_modules = (
            kind in MODULAR_PROP_KINDS
            or (kind == "sofa" and height > width)
            or (kind == "console" and height == 1 and width >= 3)
        )
        if not split_into_modules:
            return [Prop(kind, tile_x, tile_y, width, height, True)]
        return [
            Prop(kind, tile_x + dx, tile_y + dy, 1, 1, True)
            for dy in range(height)
            for dx in range(width)
        ]

    def _terminal_tile(
        self,
        room: Room,
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        *,
        index: int,
    ) -> tuple[int, int]:
        """Choose an authored terminal socket with a usable interaction pocket."""
        offsets = [(2, 2), (8, 6), (2, 6), (8, 2), (5, 4), (5, 2), (5, 6)]
        rotation = index % len(offsets)
        offsets = offsets[rotation:] + offsets[:rotation]
        door_tiles = [(int(x), int(y)) for y, x in np.argwhere(grid == Tile.DOOR)]
        ranked: list[tuple[int, int, int, tuple[int, int]]] = []
        for order, (ox, oy) in enumerate(offsets):
            tile = (room.x + ox, room.y + oy)
            if not self._walkable(grid, blocked, tile):
                continue
            open_cardinals = sum(
                self._walkable(grid, blocked, (tile[0] + dx, tile[1] + dy))
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
            )
            door_distance = min((abs(tile[0] - x) + abs(tile[1] - y) for x, y in door_tiles), default=20)
            ranked.append((open_cardinals, min(door_distance, 6), -order, tile))
        if not ranked:
            return self._clear_tile(room.center_tile, grid, blocked)
        return max(ranked)[-1]

    def _sample_security_tiles(
        self,
        rooms: list[Room],
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        count: int,
        rng: np.random.Generator,
        avoid: set[tuple[int, int]] | None = None,
        avoid_radius: int = 0,
        doors: set[tuple[int, int]] | None = None,
        prefer_perimeter: bool = False,
    ) -> list[tuple[int, int]]:
        avoid = avoid or set()
        doors = doors or set()
        priorities = [room for room in rooms[1:-1] if room.role in ("security", "server", "vault", "lab")]
        others = [room for room in rooms[1:-1] if room not in priorities]
        rng.shuffle(priorities)
        rng.shuffle(others)
        pool = priorities + others
        results: list[tuple[int, int]] = []
        used_rooms: set[int] = set()
        for pass_index in range(2):
            if len(results) >= count:
                break
            for room in pool:
                if len(results) >= count:
                    break
                if pass_index == 0 and room.room_id in used_rooms:
                    continue
                candidates = [
                    (x, y)
                    for y in range(room.y + 1, room.y + ROOM_HEIGHT_TILES)
                    for x in range(room.x + 1, room.x + ROOM_WIDTH_TILES)
                    if grid[y, x] == Tile.FLOOR
                    and (x, y) not in blocked
                    and all(abs(x - ax) + abs(y - ay) > avoid_radius for ax, ay in avoid)
                    and all(abs(x - dx) + abs(y - dy) > DOOR_SECURITY_CLEARANCE for dx, dy in doors)
                    and all(abs(x - px) + abs(y - py) > 3 for px, py in results)
                ]
                if not candidates:
                    continue
                if prefer_perimeter:
                    wall_score = {
                        tile: sum(
                            0 <= tile[1] + dy < grid.shape[0]
                            and 0 <= tile[0] + dx < grid.shape[1]
                            and grid[tile[1] + dy, tile[0] + dx] == Tile.WALL
                            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        )
                        for tile in candidates
                    }
                    best = max(wall_score.values())
                    candidates = [tile for tile in candidates if wall_score[tile] == best]
                results.append(candidates[int(rng.integers(0, len(candidates)))])
                used_rooms.add(room.room_id)
        return results

    def _camera_base_angle(
        self,
        tile: tuple[int, int],
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        objectives: set[tuple[int, int]],
        *,
        phase: float,
    ) -> float:
        origin = tile_center(tile)
        scored: list[tuple[int, float, int, float]] = []
        for index in range(8):
            angle = phase + index * math.tau / 8.0
            objective_exposure = sum(
                self._camera_can_sweep_position(grid, blocked, origin, angle, tile_center(objective))
                for objective in objectives
            )
            open_distance = self._ray_distance(grid, blocked, origin, angle, limit=CAMERA_VISION_DISTANCE)
            scored.append((-objective_exposure, open_distance, -index, angle))
        return max(scored)[-1]

    def _patrol_route(
        self,
        *,
        start: tuple[int, int],
        room_id: int,
        rooms: list[Room],
        doors: list[Door],
        adjacency: dict[int, list[int]],
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        route_index: int,
    ) -> list[tuple[int, int]]:
        room = rooms[room_id]
        local = self._room_anchor(room, grid, blocked, away_from=start, prefer_far=True)
        route = [start]
        if local != start:
            route.append(local)
        neighbours = sorted(adjacency.get(room_id, ()))
        if neighbours:
            neighbour_id = neighbours[route_index % len(neighbours)]
            doorway = next(
                door.tile for door in doors if {door.room_a, door.room_b} == {room_id, neighbour_id}
            )
            neighbour = self._room_anchor(
                rooms[neighbour_id], grid, blocked, away_from=doorway, prefer_far=False
            )
            route.extend((doorway, neighbour))
        unique: list[tuple[int, int]] = []
        for tile in route:
            if tile not in unique:
                unique.append(tile)
        if len(unique) < 3:
            fallback = self._room_anchor(room, grid, blocked, away_from=local, prefer_far=True)
            if fallback not in unique:
                unique.append(fallback)
        return unique

    def _room_anchor(
        self,
        room: Room,
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        *,
        away_from: tuple[int, int],
        prefer_far: bool,
    ) -> tuple[int, int]:
        candidates = [
            (x, y)
            for y in range(room.y + 1, room.y + ROOM_HEIGHT_TILES)
            for x in range(room.x + 1, room.x + ROOM_WIDTH_TILES)
            if self._walkable(grid, blocked, (x, y))
        ]
        if not candidates:
            return away_from

        def score(tile: tuple[int, int]) -> tuple[int, int, int, int]:
            distance = abs(tile[0] - away_from[0]) + abs(tile[1] - away_from[1])
            clearance = sum(
                self._walkable(grid, blocked, (tile[0] + dx, tile[1] + dy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            preferred_distance = distance if prefer_far else -abs(distance - 3)
            return preferred_distance, clearance, -tile[1], -tile[0]

        return max(candidates, key=score)

    @staticmethod
    def _walkable(grid: np.ndarray, blocked: set[tuple[int, int]], tile: tuple[int, int]) -> bool:
        x, y = tile
        return (
            0 <= y < grid.shape[0]
            and 0 <= x < grid.shape[1]
            and tile not in blocked
            and grid[y, x] != Tile.WALL
        )

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        return abs((first - second + math.pi) % math.tau - math.pi)

    @classmethod
    def _camera_can_sweep_position(
        cls,
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        origin: np.ndarray,
        base_angle: float,
        target: np.ndarray,
        *,
        sweep: float = CAMERA_SWEEP_RADIANS,
    ) -> bool:
        delta = target - origin
        distance = math.hypot(float(delta[0]), float(delta[1]))
        if distance > CAMERA_VISION_DISTANCE or distance <= 1e-6:
            return False
        angle = math.atan2(float(delta[1]), float(delta[0]))
        if cls._angle_difference(angle, base_angle) > sweep + CAMERA_HALF_ANGLE:
            return False
        return cls._grid_line_of_sight(grid, blocked, origin, target)

    @staticmethod
    def _grid_line_of_sight(
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        origin: np.ndarray,
        target: np.ndarray,
    ) -> bool:
        delta = target - origin
        distance = math.hypot(float(delta[0]), float(delta[1]))
        if distance <= 1e-6:
            return True
        direction = delta / distance
        for step in range(1, int(distance // 12.0) + 1):
            sample = origin + direction * (step * 12.0)
            tile = (int(sample[0] // TILE_SIZE), int(sample[1] // TILE_SIZE))
            if tile in blocked or grid[tile[1], tile[0]] == Tile.WALL:
                return False
        return True

    @staticmethod
    def _ray_distance(
        grid: np.ndarray,
        blocked: set[tuple[int, int]],
        origin: np.ndarray,
        angle: float,
        *,
        limit: float,
    ) -> float:
        direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)
        distance = 0.0
        while distance < limit:
            distance += 12.0
            sample = origin + direction * distance
            tile = (int(sample[0] // TILE_SIZE), int(sample[1] // TILE_SIZE))
            if not (0 <= tile[1] < grid.shape[0] and 0 <= tile[0] < grid.shape[1]):
                break
            if tile in blocked or grid[tile[1], tile[0]] == Tile.WALL:
                break
        return min(distance, limit)

    @staticmethod
    def _clear_tile(candidate: tuple[int, int], grid: np.ndarray, blocked: set[tuple[int, int]]) -> tuple[int, int]:
        if grid[candidate[1], candidate[0]] != Tile.WALL and candidate not in blocked:
            return candidate
        queue = deque([candidate])
        seen = {candidate}
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in seen or not (0 <= nxt[1] < grid.shape[0] and 0 <= nxt[0] < grid.shape[1]):
                    continue
                if grid[nxt[1], nxt[0]] != Tile.WALL and nxt not in blocked:
                    return nxt
                seen.add(nxt)
                queue.append(nxt)
        raise RuntimeError("level has no clear tile")

    @staticmethod
    def room_at_tile(rooms: list[Room], tile: tuple[int, int]) -> int:
        x, y = tile
        for room in rooms:
            if room.x <= x <= room.x + ROOM_WIDTH_TILES and room.y <= y <= room.y + ROOM_HEIGHT_TILES:
                return room.room_id
        return 0

    def terminal_approach_tiles(
        self,
        level: GeneratedLevel,
        terminal: Terminal,
        *,
        camera_safe_only: bool = False,
    ) -> list[tuple[int, int]]:
        """Return robust tile-centre link positions a human or agent can use.

        This is deliberately based on the same geometry as simulation LOS. It
        is useful both to validate content and to keep procedural objectives
        from becoming permanently covered by overlapping camera sweeps.
        """
        blocked = {
            (prop.tile_x + dx, prop.tile_y + dy)
            for prop in level.props
            if prop.blocking
            for dx in range(prop.width)
            for dy in range(prop.height)
        }
        terminal_tile = world_to_tile(terminal.position)
        candidates = [
            (terminal_tile[0] + dx, terminal_tile[1] + dy)
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
            if self._walkable(level.grid, blocked, (terminal_tile[0] + dx, terminal_tile[1] + dy))
            and math.hypot(dx * TILE_SIZE, dy * TILE_SIZE) <= HACK_RADIUS
        ]
        if not camera_safe_only:
            return candidates
        return [
            tile
            for tile in candidates
            if not any(
                self._camera_can_sweep_position(
                    level.grid,
                    blocked,
                    camera.position,
                    camera.base_angle,
                    tile_center(tile),
                    sweep=camera.sweep,
                )
                for camera in level.cameras
            )
        ]

    def validate(self, level: GeneratedLevel) -> bool:
        spec = TIERS[level.tier]
        if len(level.cameras) != spec.cameras or len(level.guards) != spec.guards:
            return False
        if sum(terminal.value for terminal in level.terminals) < level.quota:
            return False
        if len(level.doors) - (len(level.rooms) - 1) < MIN_GRAPH_LOOPS[level.tier]:
            return False
        blocked = {(p.tile_x + dx, p.tile_y + dy) for p in level.props if p.blocking for dx in range(p.width) for dy in range(p.height)}
        start = world_to_tile(level.spawn)
        reachable = flood_fill(level.grid, start, blocked)
        required = [world_to_tile(level.extraction), *(world_to_tile(t.position) for t in level.terminals)]
        if not all(tile in reachable for tile in required) or len(reachable) < max(12, int(np.count_nonzero(level.grid != Tile.WALL) * 0.72)):
            return False
        for door in level.doors:
            room_a, room_b = level.rooms[door.room_a], level.rooms[door.room_b]
            x, y = door.tile
            approaches = ((x - 1, y), (x + 1, y)) if room_a.row == room_b.row else ((x, y - 1), (x, y + 1))
            if not all(self._walkable(level.grid, blocked, tile) for tile in (door.tile, *approaches)):
                return False

        objectives = required + [start]
        camera_tiles = [world_to_tile(camera.position) for camera in level.cameras]
        guard_tiles = [world_to_tile(guard.position) for guard in level.guards]
        if len(set(camera_tiles + guard_tiles)) != len(camera_tiles) + len(guard_tiles):
            return False
        if any(tile not in reachable for tile in camera_tiles + guard_tiles):
            return False
        if any(
            min((abs(x - sx) + abs(y - sy) for sx, sy in camera_tiles), default=99)
            <= CAMERA_OBJECTIVE_CLEARANCE
            for x, y in objectives
        ):
            return False
        if any(
            min((abs(x - sx) + abs(y - sy) for sx, sy in guard_tiles), default=99)
            <= GUARD_OBJECTIVE_CLEARANCE
            for x, y in objectives
        ):
            return False
        if any(len(self.terminal_approach_tiles(level, terminal)) < 3 for terminal in level.terminals):
            return False
        if any(not self.terminal_approach_tiles(level, terminal, camera_safe_only=True) for terminal in level.terminals):
            return False
        if any(
            self._camera_can_sweep_position(
                level.grid,
                blocked,
                camera.position,
                camera.base_angle,
                objective,
                sweep=camera.sweep,
            )
            for camera in level.cameras
            for objective in (level.spawn, level.extraction)
        ):
            return False
        for guard in level.guards:
            patrol_tiles = [world_to_tile(point) for point in guard.patrol]
            if len(set(patrol_tiles)) < 3 or any(tile not in reachable for tile in patrol_tiles):
                return False
        return True


def world_to_tile(position: np.ndarray) -> tuple[int, int]:
    return int(position[0] // TILE_SIZE), int(position[1] // TILE_SIZE)


def flood_fill(grid: np.ndarray, start: tuple[int, int], blocked: set[tuple[int, int]] | None = None) -> set[tuple[int, int]]:
    blocked = blocked or set()
    if start in blocked or grid[start[1], start[0]] == Tile.WALL:
        return set()
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in seen or nxt in blocked or not (0 <= nxt[1] < grid.shape[0] and 0 <= nxt[0] < grid.shape[1]):
                continue
            if grid[nxt[1], nxt[0]] == Tile.WALL:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen
