"""Emit a compact action-run trace for the public-observation teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ghostline.env import GhostlineEnv
from ghostline.policies import ObservationTeacherPolicy
from ghostline.types import Action


def trace(seed: int, tier: int) -> dict[str, object]:
    env = GhostlineEnv(seed=seed, tier=tier)
    observation, _ = env.reset(seed=seed, options={"tier": tier})
    teacher = ObservationTeacherPolicy()
    steps: list[dict[str, object]] = []
    terminated = truncated = False
    while not (terminated or truncated):
        objective = np.asarray(observation["objective"], dtype=np.float32)
        action = teacher.act(observation)
        decoded = Action.decode(action)
        steps.append(
            {
                "step": len(steps),
                "position": [round(float(value), 2) for value in env.sim.player],
                "phase": "extract" if objective[0] > 0 else "acquire",
                "goal_bearing": [round(float(value), 4) for value in objective[1:3]],
                "goal_distance": round(float((objective[3] + 1.0) * 0.5), 4),
                "waypoint_bearing": [round(float(value), 4) for value in objective[4:6]],
                "action": action,
                "move": decoded.move,
                "dash": decoded.dash,
                "pulse": decoded.pulse,
                "active_link": round(float(objective[6]), 4),
                "data": int(env.sim.data),
                "stalled_steps": int(teacher.stalled_steps),
            }
        )
        teacher.observe_executed_action(action)
        observation, _, terminated, truncated, info = env.step(action)

    runs: list[dict[str, object]] = []
    start = 0
    for index in range(1, len(steps) + 1):
        if index < len(steps) and steps[index]["action"] == steps[start]["action"]:
            continue
        first, last = steps[start], steps[index - 1]
        runs.append(
            {
                "start": start,
                "end": index - 1,
                "length": index - start,
                "action": first["action"],
                "move": first["move"],
                "from": first["position"],
                "to": last["position"],
                "phase": first["phase"],
                "goal_distance_start": first["goal_distance"],
                "goal_distance_end": last["goal_distance"],
                "waypoint_start": first["waypoint_bearing"],
                "waypoint_end": last["waypoint_bearing"],
                "link_start": first["active_link"],
                "link_end": last["active_link"],
                "data_start": first["data"],
                "data_end": last["data"],
                "max_stalled": max(int(item["stalled_steps"]) for item in steps[start:index]),
            }
        )
        start = index
    compact_runs = [
        run
        for index, run in enumerate(runs)
        if index < 25
        or run["length"] >= 8
        or run["link_start"] != run["link_end"]
        or run["data_start"] != run["data_end"]
    ]
    result = {
        "seed": seed,
        "tier": tier,
        "success": bool(info["is_success"]),
        "fail_reason": str(info["fail_reason"]),
        "decisions": len(steps),
        "data": int(env.sim.data),
        "quota": int(env.sim.level.quota),
        "distance_travelled": round(float(env.sim.distance_travelled), 2),
        "action_runs": len(runs),
        "max_repeated_action": max((int(run["length"]) for run in runs), default=0),
        "compact_runs": compact_runs,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = [trace(seed, args.tier) for seed in args.seed]
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
