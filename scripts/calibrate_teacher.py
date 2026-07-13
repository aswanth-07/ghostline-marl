from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from statistics import median
from typing import Any

from ghostline.env import GhostlineEnv
from ghostline.policies import ObservationTeacherPolicy, TeacherConfig
from ghostline.seeds import validation_seed


def run_episode(job: tuple[int, int, dict[str, float]]) -> dict[str, Any]:
    tier, seed, values = job
    env = GhostlineEnv(seed=seed, tier=tier)
    observation, _ = env.reset(seed=seed)
    teacher = ObservationTeacherPolicy(TeacherConfig(**values))
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(teacher.act(observation))
    env.close()
    return info


def main() -> None:
    defaults = TeacherConfig()
    parser = argparse.ArgumentParser(description="Calibrate the fair Ghostline observation-only teacher")
    parser.add_argument("--episodes", type=int, default=20, help="Held-out episodes per tier")
    parser.add_argument("--tiers", default="3,4,5,6")
    parser.add_argument(
        "--validation-offset",
        type=int,
        default=0,
        help="per-tier offset inside the reserved validation namespace",
    )
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--objective-weight", type=float, default=defaults.objective_weight)
    parser.add_argument("--clearance-weight", type=float, default=defaults.clearance_weight)
    parser.add_argument("--escape-weight", type=float, default=defaults.escape_weight)
    parser.add_argument("--sight-weight", type=float, default=defaults.sight_weight)
    parser.add_argument("--inertia-weight", type=float, default=defaults.inertia_weight)
    parser.add_argument("--tier6-inertia-scale", type=float, default=defaults.tier6_inertia_scale)
    parser.add_argument("--dash-energy-threshold", type=float, default=defaults.dash_energy_threshold)
    parser.add_argument("--pulse-trace-threshold", type=float, default=defaults.pulse_trace_threshold)
    args = parser.parse_args()
    tiers = tuple(int(value) for value in args.tiers.split(","))
    config = TeacherConfig(
        objective_weight=args.objective_weight,
        clearance_weight=args.clearance_weight,
        escape_weight=args.escape_weight,
        sight_weight=args.sight_weight,
        inertia_weight=args.inertia_weight,
        tier6_inertia_scale=args.tier6_inertia_scale,
        dash_energy_threshold=args.dash_energy_threshold,
        pulse_trace_threshold=args.pulse_trace_threshold,
    )
    config_values = asdict(config)
    jobs = [
        (tier, validation_seed(tier, args.validation_offset + episode), config_values)
        for tier in tiers
        for episode in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        results = list(pool.map(run_episode, jobs))
    report: dict[str, Any] = {"config": config_values, "episodes_per_tier": args.episodes, "tiers": {}}
    for tier in tiers:
        rows = [row for row in results if int(row["tier"]) == tier]
        successes = sum(bool(row["is_success"]) for row in rows)
        report["tiers"][str(tier)] = {
            "successes": successes,
            "success_rate": successes / max(1, len(rows)),
            "fail_reasons": dict(Counter(str(row["fail_reason"]) for row in rows)),
            "median_duration_seconds": median(float(row["duration_seconds"]) for row in rows),
            "mean_damage": sum(int(row["damage"]) for row in rows) / max(1, len(rows)),
            "mean_detections": sum(int(row["detections"]) for row in rows) / max(1, len(rows)),
            "mean_pulse_uses": sum(int(row["pulse_uses"]) for row in rows) / max(1, len(rows)),
            "failure_samples": [
                {
                    "seed": int(row["seed"]),
                    "reason": str(row["fail_reason"]),
                    "data": int(row["data"]),
                    "quota": int(row["quota"]),
                    "duration_seconds": round(float(row["duration_seconds"]), 3),
                    "damage": int(row["damage"]),
                    "detections": int(row["detections"]),
                    "max_trace": round(float(row["max_trace"]), 3),
                }
                for row in rows
                if not bool(row["is_success"])
            ][:10],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
