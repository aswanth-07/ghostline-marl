from __future__ import annotations

import argparse
import time

from ghostline.generation import LevelGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10_000)
    args = parser.parse_args()
    generator = LevelGenerator()
    started = time.perf_counter()
    generated = 0
    for seed in range(args.seeds):
        tier = seed % 6 + 1
        level = generator.generate(seed=seed, tier=tier)
        if not generator.validate(level):
            raise RuntimeError(f"invalid level: seed={seed} tier={tier}")
        generated += 1
    elapsed = time.perf_counter() - started
    print(f"validated {generated} levels in {elapsed:.2f}s ({generated / elapsed:.1f} levels/s)")


if __name__ == "__main__":
    main()
