"""Measure and record deterministic Ghostline headless-simulation throughput."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Sequence

import numpy as np

from ghostline.env import GhostlineEnv
from ghostline.onnx_contract import environment_fingerprint


ROOT = Path(__file__).resolve().parents[1]
REPORT_CONTRACT = "ghostline-headless-throughput-v1"
OBSERVATION_CONTRACT = "GhostlineEnv-v2"
ACTION_REPEAT_TICKS = 6


def run_worker(worker: int, decisions: int, tier: int) -> tuple[float, int]:
    """Run one independent deterministic random-action workload."""

    env = GhostlineEnv(seed=worker, tier=tier)
    env.reset(seed=worker)
    rng = np.random.default_rng(9_173 + worker)
    resets = 0
    started = time.perf_counter()
    try:
        for _ in range(decisions):
            _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 36)))
            if terminated or truncated:
                resets += 1
                env.reset()
    finally:
        env.close()
    return time.perf_counter() - started, resets


def _build_report(
    *,
    decisions: int,
    tier: int,
    workers: int,
    wall_elapsed: float,
    results: Sequence[tuple[float, int]],
    minimum_decisions_per_second: float,
    fingerprint: str,
) -> dict[str, object]:
    """Build the machine-readable report from raw worker timings."""

    if decisions <= 0:
        raise ValueError("decisions must be positive")
    if tier not in range(1, 7):
        raise ValueError("tier must lie in 1..6")
    if workers <= 0 or len(results) != workers:
        raise ValueError("results must contain exactly one record per worker")
    if wall_elapsed <= 0 or any(elapsed <= 0 for elapsed, _ in results):
        raise ValueError("benchmark timings must be positive")
    if minimum_decisions_per_second < 0:
        raise ValueError("minimum throughput must be non-negative")

    total = decisions * workers
    aggregate_rate = total / wall_elapsed
    worker_rates = [decisions / elapsed for elapsed, _ in results]
    return {
        "report_contract": REPORT_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "environment_fingerprint": fingerprint,
        "tier": tier,
        "action_repeat_ticks": ACTION_REPEAT_TICKS,
        "decisions_per_worker": decisions,
        "workers": workers,
        "total_decisions": total,
        "wall_elapsed_seconds": wall_elapsed,
        "worker_elapsed_seconds": [elapsed for elapsed, _ in results],
        "aggregate_decisions_per_second": aggregate_rate,
        "aggregate_simulation_ticks_per_second": aggregate_rate * ACTION_REPEAT_TICKS,
        "median_worker_decisions_per_second": float(np.median(worker_rates)),
        "resets": sum(resets for _, resets in results),
        "minimum_decisions_per_second": minimum_decisions_per_second,
        "meets_minimum": aggregate_rate >= minimum_decisions_per_second,
        "system": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }


def run_benchmark(
    *,
    decisions: int,
    tier: int,
    workers: int,
    minimum_decisions_per_second: float = 0.0,
) -> dict[str, object]:
    """Execute the benchmark and return its provenance-bound report."""

    if decisions <= 0:
        raise ValueError("decisions must be positive")
    if tier not in range(1, 7):
        raise ValueError("tier must lie in 1..6")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if minimum_decisions_per_second < 0:
        raise ValueError("minimum throughput must be non-negative")
    workers = min(workers, os.cpu_count() or workers)
    wall_started = time.perf_counter()
    if workers == 1:
        results = [run_worker(0, decisions, tier)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    run_worker,
                    range(workers),
                    [decisions] * workers,
                    [tier] * workers,
                )
            )
    wall_elapsed = time.perf_counter() - wall_started
    return _build_report(
        decisions=decisions,
        tier=tier,
        workers=workers,
        wall_elapsed=wall_elapsed,
        results=results,
        minimum_decisions_per_second=minimum_decisions_per_second,
        fingerprint=environment_fingerprint(ROOT / "src" / "ghostline"),
    )


def _stable_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=2_000, help="Decisions per worker")
    parser.add_argument("--tier", type=int, choices=range(1, 7), default=6)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent headless processes; decisions are measured per worker.",
    )
    parser.add_argument(
        "--minimum-decisions-per-second",
        type=float,
        default=0.0,
        help="Exit unsuccessfully if aggregate throughput is below this value.",
    )
    parser.add_argument("--output", type=Path, help="Optional machine-readable JSON report")
    args = parser.parse_args()
    try:
        report = run_benchmark(
            decisions=args.decisions,
            tier=args.tier,
            workers=args.workers,
            minimum_decisions_per_second=args.minimum_decisions_per_second,
        )
    except ValueError as error:
        parser.error(str(error))
    serialized = _stable_json(report)
    if args.output is not None:
        if args.output.suffix.casefold() != ".json":
            parser.error("--output must use a .json suffix")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")
    if not report["meets_minimum"]:
        print(
            "error: aggregate headless throughput did not meet the requested minimum",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
