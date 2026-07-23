from __future__ import annotations

import argparse
import time

from ghostline.generation import LevelGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10_000)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="also audit Env-v3 security-door selection and directive metadata",
    )
    args = parser.parse_args()
    generator = LevelGenerator()
    started = time.perf_counter()
    generated = 0
    for seed in range(args.seeds):
        tier = seed % 6 + 1
        sim = None
        if args.adaptive:
            from ghostline.simulation_v3 import GhostlineSimulationV3

            sim = GhostlineSimulationV3(seed=seed, tier=tier)
            level = sim.level
        else:
            level = generator.generate(seed=seed, tier=tier)
        if not generator.validate(level):
            raise RuntimeError(f"invalid level: seed={seed} tier={tier}")
        if sim is not None:
            expected = {4: 1, 5: 2, 6: 3}.get(tier, 0)
            if len(sim.security_doors) != expected:
                raise RuntimeError(
                    f"wrong adaptive-door count: seed={seed} tier={tier} "
                    f"expected={expected} actual={len(sim.security_doors)}"
                )
            doors_by_tile = {door.tile: door for door in level.doors}
            if len(doors_by_tile) != len(level.doors):
                raise RuntimeError(f"duplicate generated door tile: seed={seed} tier={tier}")
            for security_door in sim.security_doors:
                source = doors_by_tile[security_door.tile]
                if not sim._door_edge_is_redundant(source.room_a, source.room_b):
                    raise RuntimeError(
                        f"adaptive lock selected a bridge edge: seed={seed} tier={tier} "
                        f"tile={security_door.tile}"
                    )
            if sim.directive_par_seconds <= 0.0:
                raise RuntimeError(f"invalid adaptive speed par: seed={seed} tier={tier}")
        generated += 1
    elapsed = time.perf_counter() - started
    print(f"validated {generated} levels in {elapsed:.2f}s ({generated / elapsed:.1f} levels/s)")


if __name__ == "__main__":
    main()
