from __future__ import annotations

import argparse
import copy
import time

from ghostline.generation import LevelGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10_000)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="audit the developmental multi-agent v2 layout and field-system contract",
    )
    args = parser.parse_args()
    generator = LevelGenerator()
    started = time.perf_counter()
    generated = 0
    for seed in range(args.seeds):
        tier = seed % 6 + 1
        sim = None
        if args.adaptive:
            from ghostline.simulation_v2 import GhostlineSimulationV2

            sim = GhostlineSimulationV2(seed=seed, tier=tier)
            validator = sim.generator
            # Runtime binding replaces a door panel's generation-time door
            # index with the corresponding security-door id. Validate a copy
            # restored to the generator representation so the 10k audit pays
            # for only one deterministic generation.
            level = copy.deepcopy(sim.level)
            door_index_by_tile = {
                tuple(door.tile): index for index, door in enumerate(level.doors)
            }
            for device in getattr(level, "hackable", ()):
                if device.kind == "door" and device.target_tile in door_index_by_tile:
                    device.target_id = door_index_by_tile[device.target_tile]
        else:
            validator = generator
            level = generator.generate(seed=seed, tier=tier)
        if sim is not None:
            readiness_errors = tuple(validator.readiness_errors(level))
            if readiness_errors:
                raise RuntimeError(
                    f"v2 content-readiness failure: seed={seed} tier={tier} "
                    f"errors={','.join(readiness_errors)}"
                )
            expected = {4: 1, 5: 2, 6: 3}.get(tier, 0)
            if len(sim.security_doors) != expected:
                raise RuntimeError(
                    f"wrong v2 security-door count: seed={seed} tier={tier} "
                    f"expected={expected} actual={len(sim.security_doors)}"
                )
            doors_by_tile = {door.tile: door for door in level.doors}
            if len(doors_by_tile) != len(level.doors):
                raise RuntimeError(f"duplicate generated door tile: seed={seed} tier={tier}")
            for security_door in sim.security_doors:
                source = doors_by_tile[security_door.tile]
                if not sim._door_edge_is_redundant(source.room_a, source.room_b):
                    raise RuntimeError(
                        f"v2 lock selected a bridge edge: seed={seed} tier={tier} "
                        f"tile={security_door.tile}"
                    )
            if sim.directive_par_seconds <= 0.0:
                raise RuntimeError(f"invalid v2 speed par: seed={seed} tier={tier}")
        elif not validator.validate(level):
            raise RuntimeError(f"invalid level: seed={seed} tier={tier}")
        generated += 1
    elapsed = time.perf_counter() - started
    print(f"validated {generated} levels in {elapsed:.2f}s ({generated / elapsed:.1f} levels/s)")


if __name__ == "__main__":
    main()
