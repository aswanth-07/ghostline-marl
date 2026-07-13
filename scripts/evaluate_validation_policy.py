"""Evaluate one neural policy on the reserved Ghostline validation namespace.

This tool is intentionally separate from final-test evaluation.  It exposes no
generic seed argument, accepts only current-fingerprint PyTorch checkpoints,
and derives every episode seed through :func:`ghostline.seeds.validation_seed`.
The resulting JSON/CSV pair is suitable for deterministic BC/DAgger/PPO
checkpoint selection without opening a final-test slice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import fmean, median
import tempfile
import time
from typing import Any, Iterable, Sequence

import torch

from ghostline.env import GhostlineEnv
from ghostline.evaluation import wilson
from ghostline.model import current_environment_fingerprint, load_policy
from ghostline.seeds import (
    FINAL_TEST_SEED_START,
    VALIDATION_SEED_END,
    VALIDATION_SEED_START,
    VALIDATION_TIER_STRIDE,
    validation_seed,
)


ALL_TIERS = (1, 2, 3, 4, 5, 6)
_WORKER_POLICY = None


def checkpoint_sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint without loading a second copy into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def build_validation_jobs(
    *,
    episodes: int,
    validation_offset: int,
    tiers: Sequence[int] = ALL_TIERS,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return ``(tier, ordinal, validation_index, seed)`` jobs in stable order.

    Duplicate tiers are rejected as seed-window overlap rather than silently
    evaluating the same contracts twice.  The explicit namespace checks are
    retained even though ``validation_seed`` also fails closed; together they
    protect this selection tool if either schedule is changed later.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if validation_offset < 0:
        raise ValueError("validation_offset must be non-negative")
    if validation_offset + episodes > VALIDATION_TIER_STRIDE:
        raise ValueError(
            "validation window leaves its per-tier block: "
            f"offset {validation_offset} + {episodes} episodes exceeds "
            f"{VALIDATION_TIER_STRIDE}"
        )

    tier_values = tuple(int(tier) for tier in tiers)
    if not tier_values:
        raise ValueError("at least one tier is required")
    if any(tier not in ALL_TIERS for tier in tier_values):
        raise ValueError("tiers must lie in 1..6")
    if len(set(tier_values)) != len(tier_values):
        raise ValueError("validation seed overlap: duplicate tiers are not allowed")

    jobs: list[tuple[int, int, int, int]] = []
    seen_seeds: set[int] = set()
    for tier in sorted(tier_values):
        for ordinal in range(episodes):
            validation_index = validation_offset + ordinal
            seed = validation_seed(tier, validation_index)
            if not VALIDATION_SEED_START <= seed <= VALIDATION_SEED_END:
                raise ValueError(f"seed {seed} left the reserved validation namespace")
            if seed >= FINAL_TEST_SEED_START:
                raise ValueError(f"final-test seed {seed} is forbidden during checkpoint selection")
            if seed in seen_seeds:
                raise ValueError(f"validation seed overlap detected at {seed}")
            seen_seeds.add(seed)
            jobs.append((tier, ordinal, validation_index, seed))
    return tuple(jobs)


def _configure_cpu_worker() -> None:
    """Keep each inference worker deterministic and free of thread oversubscription."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits configuring the inter-op pool only before it starts.
        # The intra-op setting above is the authoritative worker requirement.
        pass
    torch.use_deterministic_algorithms(True)


def _init_policy_worker(checkpoint: str) -> None:
    global _WORKER_POLICY
    _configure_cpu_worker()
    _WORKER_POLICY = load_policy(Path(checkpoint), device="cpu")


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    location = (len(ordered) - 1) * quantile
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _model_episode(job: tuple[int, int, int, int]) -> dict[str, Any]:
    tier, ordinal, validation_index, seed = job
    if _WORKER_POLICY is None:
        raise RuntimeError("validation worker was not initialized with a neural policy")

    env = GhostlineEnv(seed=seed, tier=tier)
    action_digest = hashlib.sha256()
    latencies_ms: list[float] = []
    info: dict[str, Any] | None = None
    try:
        observation, _ = env.reset(seed=seed)
        hidden = None
        terminated = truncated = False
        while not (terminated or truncated):
            started_ns = time.perf_counter_ns()
            action, hidden = _WORKER_POLICY.act(
                observation,
                hidden,
                deterministic=True,
                device="cpu",
            )
            latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
            action_digest.update(bytes((int(action),)))
            observation, _, terminated, truncated, info = env.step(action)

        if info is None:
            raise RuntimeError(f"tier {tier} seed {seed} terminated without terminal info")
        if int(info["tier"]) != tier or int(info["seed"]) != seed:
            raise RuntimeError(
                f"terminal identity mismatch for requested tier {tier}, seed {seed}: "
                f"received tier {info['tier']}, seed {info['seed']}"
            )
        telemetry = info.get("telemetry")
        if not isinstance(telemetry, dict) or "path_efficiency" not in telemetry:
            raise RuntimeError(f"tier {tier} seed {seed} omitted terminal path efficiency")

        return {
            "tier": tier,
            "episode_ordinal": ordinal,
            "validation_index": validation_index,
            "seed": seed,
            "success": bool(info["is_success"]),
            "failure_reason": str(info["fail_reason"]),
            "damage": int(info["damage"]),
            "detections": int(info["detections"]),
            "duration_seconds": float(info["duration_seconds"]),
            "path_efficiency": float(telemetry["path_efficiency"]),
            "decision_count": len(latencies_ms),
            "policy_latency_total_ms": float(sum(latencies_ms)),
            "mean_policy_latency_ms": float(fmean(latencies_ms)),
            "median_policy_latency_ms": float(median(latencies_ms)),
            "p95_policy_latency_ms": float(_percentile(latencies_ms, 0.95)),
            "action_sha256": action_digest.hexdigest(),
        }
    finally:
        env.close()


def _run_jobs(
    checkpoint: Path,
    jobs: Sequence[tuple[int, int, int, int]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    worker_count = min(int(workers), len(jobs))
    chunksize = max(1, len(jobs) // max(1, worker_count * 8))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_policy_worker,
        initargs=(str(checkpoint.resolve()),),
    ) as pool:
        # executor.map retains input ordering; aggregation also sorts and checks
        # identities so alternate executor scheduling cannot affect the report.
        return list(pool.map(_model_episode, jobs, chunksize=chunksize))


def _validate_episode_results(
    jobs: Sequence[tuple[int, int, int, int]],
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {(tier, validation_index, seed) for tier, _, validation_index, seed in jobs}
    actual = [
        (int(row["tier"]), int(row["validation_index"]), int(row["seed"]))
        for row in results
    ]
    if len(actual) != len(set(actual)):
        raise RuntimeError("worker results contain duplicate validation episodes")
    if set(actual) != expected:
        missing = sorted(expected - set(actual))[:3]
        unexpected = sorted(set(actual) - expected)[:3]
        raise RuntimeError(
            f"worker result identity mismatch; missing={missing}, unexpected={unexpected}"
        )
    for tier, validation_index, seed in actual:
        if seed != validation_seed(tier, validation_index):
            raise RuntimeError(
                f"worker returned non-validation seed {seed} for tier {tier}, index {validation_index}"
            )
        if seed >= FINAL_TEST_SEED_START:
            raise RuntimeError(f"worker returned forbidden final-test seed {seed}")
    return sorted(
        (dict(row) for row in results),
        key=lambda row: (int(row["tier"]), int(row["validation_index"]), int(row["seed"])),
    )


def _aggregate_tier(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(bool(row["success"])) for row in rows)
    interval = wilson(successes, len(rows))
    failures = Counter(
        str(row["failure_reason"])
        for row in rows
        if not bool(row["success"])
    )
    decisions = sum(int(row["decision_count"]) for row in rows)
    latency_total = sum(float(row["policy_latency_total_ms"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95": [float(interval[0]), float(interval[1])],
        "failure_reasons": {key: failures[key] for key in sorted(failures)},
        "mean_damage": float(fmean(float(row["damage"]) for row in rows)),
        "mean_detections": float(fmean(float(row["detections"]) for row in rows)),
        "median_duration_seconds": float(median(float(row["duration_seconds"]) for row in rows)),
        "mean_path_efficiency": float(fmean(float(row["path_efficiency"]) for row in rows)),
        "decision_count": decisions,
        "mean_policy_latency_ms": latency_total / max(1, decisions),
        "median_episode_policy_latency_ms": float(
            median(float(row["median_policy_latency_ms"]) for row in rows)
        ),
        "p95_episode_mean_policy_latency_ms": float(
            _percentile((float(row["mean_policy_latency_ms"]) for row in rows), 0.95)
        ),
        "episode_records": list(rows),
    }


def build_report(
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    environment_fingerprint: str,
    episodes: int,
    validation_offset: int,
    workers: int,
    jobs: Sequence[tuple[int, int, int, int]],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ordered = _validate_episode_results(jobs, results)
    tiers: dict[str, dict[str, Any]] = {}
    for tier in ALL_TIERS:
        rows = [row for row in ordered if int(row["tier"]) == tier]
        if len(rows) != episodes:
            raise RuntimeError(f"tier {tier} produced {len(rows)} results; expected {episodes}")
        tiers[str(tier)] = _aggregate_tier(rows)

    worst_tier = min(
        ALL_TIERS,
        key=lambda tier: (float(tiers[str(tier)]["success_rate"]), tier),
    )
    all_rows = [row for tier in ALL_TIERS for row in tiers[str(tier)]["episode_records"]]
    total_decisions = sum(int(row["decision_count"]) for row in all_rows)
    total_latency = sum(float(row["policy_latency_total_ms"]) for row in all_rows)
    overall_damage = float(fmean(float(row["damage"]) for row in all_rows))
    overall_efficiency = float(fmean(float(row["path_efficiency"]) for row in all_rows))
    overall_duration = float(median(float(row["duration_seconds"]) for row in all_rows))
    overall_latency = total_latency / max(1, total_decisions)

    # Values are ordered for lexicographic maximization.  Negative values make
    # lower damage, duration, and inference cost rank ahead only after the
    # success, Tier-6, and route-efficiency criteria tie.
    selection_fields = [
        "worst_tier_success_rate",
        "tier_6_success_rate",
        "negative_mean_damage",
        "mean_path_efficiency",
        "negative_median_duration_seconds",
        "negative_mean_policy_latency_ms",
    ]
    selection_values = [
        float(tiers[str(worst_tier)]["success_rate"]),
        float(tiers["6"]["success_rate"]),
        -overall_damage,
        overall_efficiency,
        -overall_duration,
        -overall_latency,
    ]
    thresholds = {str(tier): (0.85 if tier == 6 else 0.95) for tier in ALL_TIERS}

    return {
        "report_contract": "ghostline-validation-policy-v1",
        "selection_only": True,
        "final_test_seeds_used": False,
        "seed_namespace": "validation_1_000_000_to_1_049_999",
        "seed_formula": "validation_seed(tier, validation_offset + episode)",
        "validation_seed_bounds": [VALIDATION_SEED_START, VALIDATION_SEED_END],
        "validation_offset": validation_offset,
        "episodes_per_tier": episodes,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "environment_fingerprint": environment_fingerprint,
        "deterministic_actions": True,
        "inference_device": "cpu",
        "torch_version": str(torch.__version__),
        "worker_count": min(workers, len(jobs)),
        "torch_threads_per_worker": 1,
        "tiers": tiers,
        "acceptance_thresholds": thresholds,
        "meets_single_window_thresholds": all(
            float(tiers[str(tier)]["success_rate"]) >= thresholds[str(tier)]
            for tier in ALL_TIERS
        ),
        "consecutive_windows_required_for_acceptance": 2,
        "release_eligible": False,
        "worst_tier": worst_tier,
        "worst_tier_selection_tuple": selection_values,
        "selection_tuple_fields": selection_fields,
        "selection_order": "lexicographic_maximize",
    }


def _report_csv(report: dict[str, Any]) -> str:
    fields = (
        "environment_fingerprint",
        "checkpoint_sha256",
        "validation_offset",
        "tier",
        "episodes",
        "successes",
        "success_rate",
        "wilson_low",
        "wilson_high",
        "failure_reasons_json",
        "mean_damage",
        "mean_detections",
        "median_duration_seconds",
        "mean_path_efficiency",
        "mean_policy_latency_ms",
        "median_episode_policy_latency_ms",
        "p95_episode_mean_policy_latency_ms",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for tier in ALL_TIERS:
        row = report["tiers"][str(tier)]
        writer.writerow(
            {
                "environment_fingerprint": report["environment_fingerprint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "validation_offset": report["validation_offset"],
                "tier": tier,
                "episodes": row["episodes"],
                "successes": row["successes"],
                "success_rate": row["success_rate"],
                "wilson_low": row["wilson_95"][0],
                "wilson_high": row["wilson_95"][1],
                "failure_reasons_json": json.dumps(row["failure_reasons"], sort_keys=True),
                "mean_damage": row["mean_damage"],
                "mean_detections": row["mean_detections"],
                "median_duration_seconds": row["median_duration_seconds"],
                "mean_path_efficiency": row["mean_path_efficiency"],
                "mean_policy_latency_ms": row["mean_policy_latency_ms"],
                "median_episode_policy_latency_ms": row["median_episode_policy_latency_ms"],
                "p95_episode_mean_policy_latency_ms": row[
                    "p95_episode_mean_policy_latency_ms"
                ],
            }
        )
    return output.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_name = handle.name
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def evaluate_validation_policy(
    *,
    checkpoint: Path,
    episodes: int,
    validation_offset: int,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    output = Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if output.suffix.lower() != ".json":
        raise ValueError("output must use a .json suffix; a sibling .csv is written automatically")
    if workers <= 0:
        raise ValueError("workers must be positive")

    jobs = build_validation_jobs(
        episodes=episodes,
        validation_offset=validation_offset,
        tiers=ALL_TIERS,
    )
    fingerprint_before = current_environment_fingerprint()
    checkpoint_hash_before = checkpoint_sha256(checkpoint)

    # Fail fast in the parent process before starting workers.  load_policy is
    # the single checkpoint gate and rejects legacy/missing/stale fingerprints.
    policy = load_policy(checkpoint, device="cpu")
    del policy

    results = _run_jobs(checkpoint, jobs, workers=workers)
    fingerprint_after = current_environment_fingerprint()
    checkpoint_hash_after = checkpoint_sha256(checkpoint)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "environment fingerprint changed during validation; no selection report was published"
        )
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError(
            "checkpoint changed during validation; no selection report was published"
        )

    report = build_report(
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash_before,
        environment_fingerprint=fingerprint_before,
        episodes=episodes,
        validation_offset=validation_offset,
        workers=workers,
        jobs=jobs,
        results=results,
    )
    csv_path = output.with_suffix(".csv")
    _atomic_write(csv_path, _report_csv(report))
    _atomic_write(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 4) - 2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically evaluate a current Ghostline neural checkpoint on "
            "all six reserved validation-tier blocks (never final-test seeds)."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100, help="episodes per tier")
    parser.add_argument(
        "--validation-offset",
        type=int,
        required=True,
        help=f"explicit offset inside each {VALIDATION_TIER_STRIDE}-seed tier block",
    )
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--output", type=Path, required=True, help="JSON path; CSV is written beside it")
    args = parser.parse_args()

    report = evaluate_validation_policy(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        validation_offset=args.validation_offset,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "csv": str(args.output.with_suffix(".csv").resolve()),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "environment_fingerprint": report["environment_fingerprint"],
                "worst_tier": report["worst_tier"],
                "worst_tier_selection_tuple": report["worst_tier_selection_tuple"],
                "meets_single_window_thresholds": report["meets_single_window_thresholds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
