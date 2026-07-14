"""Run an explicitly non-final teacher diagnostic inside the validation namespace."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean, median

from ghostline.evaluation import _baseline_episode, wilson
from ghostline.imitation import training_environment_fingerprint
from ghostline.seeds import VALIDATION_SEED_END, VALIDATION_SEED_START


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not VALIDATION_SEED_START <= args.seed_start <= VALIDATION_SEED_END:
        raise ValueError("diagnostic seed start must lie in the validation namespace")
    if args.seed_start + args.episodes - 1 > VALIDATION_SEED_END:
        raise ValueError("diagnostic episodes leave the validation namespace")
    if not 1 <= args.workers <= 12:
        raise ValueError("workers must lie in 1..12")

    fingerprint_start = training_environment_fingerprint()
    if fingerprint_start != args.expected_fingerprint:
        raise RuntimeError(
            f"environment fingerprint mismatch before diagnostic: {fingerprint_start}"
        )
    report: dict[str, object] = {
        "diagnostic_only": True,
        "seed_namespace": "validation_retired",
        "seed_start": args.seed_start,
        "episodes_per_tier": args.episodes,
        "environment_fingerprint": fingerprint_start,
        "tiers": {},
    }
    for tier in range(1, 7):
        jobs = [
            (tier, args.seed_start + episode, episode, "teacher")
            for episode in range(args.episodes)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            infos = list(pool.map(_baseline_episode, jobs))
        successes = sum(int(info["success"]) for info in infos)
        failures = Counter(str(info["failure_reason"]) for info in infos)
        damage = [int(info["damage"]) for info in infos]
        report["tiers"][str(tier)] = {
            "successes": successes,
            "success_rate": successes / args.episodes,
            "wilson_95": list(wilson(successes, args.episodes)),
            "fail_reasons": dict(failures),
            "mean_damage": mean(damage),
            "mean_guard_damage": mean(int(info.get("damage_by_guard", 0)) for info in infos),
            "mean_drone_damage": mean(int(info.get("damage_by_drone", 0)) for info in infos),
            "median_duration_seconds": median(float(info["duration_seconds"]) for info in infos),
            "median_max_trace": median(float(info["max_trace"]) for info in infos),
            "episodes": [
                {
                    "seed": seed,
                    "success": bool(info["success"]),
                    "fail_reason": str(info["failure_reason"]),
                    "damage": int(info["damage"]),
                    "duration_seconds": float(info["duration_seconds"]),
                }
                for (_, seed, _, _), info in zip(jobs, infos, strict=True)
            ],
        }

    fingerprint_end = training_environment_fingerprint()
    if fingerprint_end != fingerprint_start:
        raise RuntimeError(
            "environment fingerprint changed during diagnostic; no report was published"
        )
    report["environment_stable_during_diagnostic"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
