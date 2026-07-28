"""Held-out evaluation for the developmental 288-action Ghostline runner."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, median
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ghostline.curriculum import ACCEPTANCE_THRESHOLDS
from ghostline.env_v2 import GhostlineEnvV2
from ghostline.model_v2 import (
    OBSERVATION_CONTRACT_V2,
    RunnerPolicyV2,
    load_runner_v2,
    multi_agent_environment_fingerprint,
    runner_model_fingerprint,
)
from ghostline.seeds import FINAL_TEST_SEED_START, final_test_seed
from ghostline.types_v2 import ContractDirective


EVALUATION_CONTRACT_V2 = "ghostline-runner-final-evaluation-v2.0"
DEFAULT_V2_SLICE_MANIFEST = Path(
    "benchmarks/runner-v2/final-test-slices.json"
)
ALL_TIERS = (1, 2, 3, 4, 5, 6)
ALL_DIRECTIVES = (
    ContractDirective.STANDARD,
    ContractDirective.GHOST,
    ContractDirective.SPEED,
    ContractDirective.GREED,
)

_WORKER_POLICY: RunnerPolicyV2 | None = None
_WORKER_SECURITY_POOL: Any | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one episode")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            (
                proportion * (1.0 - proportion)
                + z * z / (4.0 * total)
            )
            / total
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _parse_tiers(values: str | Iterable[int]) -> tuple[int, ...]:
    tiers = (
        tuple(int(value.strip()) for value in values.split(",") if value.strip())
        if isinstance(values, str)
        else tuple(int(value) for value in values)
    )
    if (
        not tiers
        or len(set(tiers)) != len(tiers)
        or any(tier not in ALL_TIERS for tier in tiers)
    ):
        raise ValueError("tiers must be a unique non-empty subset of 1..6")
    return tiers


def _parse_directives(
    values: str | Iterable[ContractDirective | str | int],
) -> tuple[ContractDirective, ...]:
    raw: Iterable[ContractDirective | str | int] = (
        (value.strip() for value in values.split(",") if value.strip())
        if isinstance(values, str)
        else values
    )
    directives = tuple(ContractDirective.parse(value) for value in raw)
    if not directives or len(set(directives)) != len(directives):
        raise ValueError("directives must be a unique non-empty list")
    return directives


def _init_worker(
    checkpoint: str,
    security_checkpoints: tuple[str, ...],
    security_pool_salt: int,
) -> None:
    global _WORKER_POLICY, _WORKER_SECURITY_POOL
    torch.set_num_threads(1)
    _WORKER_POLICY = load_runner_v2(Path(checkpoint), device="cpu")
    if security_checkpoints:
        from ghostline.security_opponents import FrozenSecurityOpponentPool

        _WORKER_SECURITY_POOL = FrozenSecurityOpponentPool(
            tuple(Path(path) for path in security_checkpoints),
            selection_salt=security_pool_salt,
        )
    else:
        _WORKER_SECURITY_POOL = None


def _action_digest(actions: Sequence[int]) -> str:
    values = np.asarray(actions, dtype="<u2")
    if np.any(values >= 288):
        raise RuntimeError("evaluation recorded an out-of-contract v2 action")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _episode(task: tuple[int, int, int, int]) -> dict[str, Any]:
    tier, directive_value, episode_index, seed = task
    if _WORKER_POLICY is None:
        raise RuntimeError("v2 evaluation worker policy was not initialized")
    directive = ContractDirective(directive_value)
    if _WORKER_SECURITY_POOL is None:
        env: GhostlineEnvV2 = GhostlineEnvV2(
            seed=seed,
            tier=tier,
            directive=directive,
        )
    else:
        from ghostline.security_opponents import FrozenSecurityRunnerEnvV2

        env = FrozenSecurityRunnerEnvV2(
            security_pool=_WORKER_SECURITY_POOL,
            seed=seed,
            tier=tier,
            directive=directive,
        )
    observation, _ = env.reset(
        seed=seed,
        options={"tier": tier, "directive": directive},
    )
    hidden: torch.Tensor | None = None
    actions: list[int] = []
    latencies: list[float] = []
    terminated = truncated = False
    info: dict[str, Any] = {}
    try:
        while not (terminated or truncated):
            started = time.perf_counter()
            action, hidden = _WORKER_POLICY.act(
                observation,
                hidden,
                deterministic=True,
                device="cpu",
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            if not bool(observation["action_mask"][action]):
                raise RuntimeError("v2 policy selected an action masked by the environment")
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
        telemetry = info.get("telemetry")
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        rewards = info.get("reward_components")
        rewards = rewards if isinstance(rewards, dict) else {}
        if not rewards:
            rewards = {
                key.removeprefix("reward_"): value
                for key, value in info.items()
                if key.startswith("reward_")
                and key != "reward_total"
                and isinstance(value, (int, float))
            }
        reward_components = {
            str(name): float(value)
            for name, value in sorted(rewards.items())
            if isinstance(value, (int, float))
        }
        record = {
            "tier": tier,
            "directive": directive.name.lower(),
            "episode_index": episode_index,
            "seed": seed,
            "success": bool(info.get("is_success", False)),
            "failure_reason": str(info.get("fail_reason", "unknown")),
            "duration_seconds": float(
                info.get("duration_seconds", env.sim.elapsed_seconds)
            ),
            "max_trace": float(info.get("max_trace", env.sim.max_trace)),
            "damage": int(info.get("damage", env.sim.damage_taken)),
            "detections": int(info.get("detections", env.sim.detections)),
            "optional_data": int(info.get("optional_data", 0)),
            "path_efficiency": (
                float(telemetry["path_efficiency"])
                if telemetry.get("path_efficiency") is not None
                else None
            ),
            "decision_count": len(actions),
            "action_sha256": _action_digest(actions),
            "median_policy_latency_ms": float(median(latencies)),
            "reward_total": float(sum(reward_components.values())),
            "reward_components": reward_components,
            "security_opponent_id": info.get("security_opponent_id"),
        }
        declared_total = info.get("reward_total")
        if isinstance(declared_total, (int, float)) and not math.isclose(
            float(declared_total),
            record["reward_total"],
            abs_tol=1e-6,
        ):
            raise RuntimeError("v2 terminal reward ledger does not sum exactly")
        return record
    finally:
        env.close()


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(record["success"]) for record in records)
    low, high = _wilson(successes, len(records))
    path_values = [
        float(record["path_efficiency"])
        for record in records
        if record["path_efficiency"] is not None
    ]
    failures: dict[str, int] = defaultdict(int)
    for record in records:
        if not record["success"]:
            failures[str(record["failure_reason"])] += 1
    return {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "success_ci95_low": low,
        "success_ci95_high": high,
        "mean_damage": fmean(float(record["damage"]) for record in records),
        "mean_max_trace": fmean(float(record["max_trace"]) for record in records),
        "mean_detections": fmean(
            float(record["detections"]) for record in records
        ),
        "median_duration_seconds": median(
            float(record["duration_seconds"]) for record in records
        ),
        "mean_path_efficiency": (
            fmean(path_values) if path_values else None
        ),
        "median_policy_latency_ms": median(
            float(record["median_policy_latency_ms"])
            for record in records
        ),
        "failure_reasons": dict(sorted(failures.items())),
    }


def _write_reports(
    output: Path,
    report: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    summary_path = output.with_suffix(".csv")
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "tier",
            "directive",
            "episodes",
            "successes",
            "success_rate",
            "success_ci95_low",
            "success_ci95_high",
            "mean_damage",
            "mean_max_trace",
            "mean_detections",
            "median_duration_seconds",
            "mean_path_efficiency",
            "median_policy_latency_ms",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for directive, tiers in report["directives"].items():
            for tier, summary in tiers.items():
                writer.writerow(
                    {
                        "tier": tier,
                        "directive": directive,
                        **{
                            key: summary[key]
                            for key in fieldnames
                            if key not in ("tier", "directive")
                        },
                    }
                )
    episode_path = output.with_name(f"{output.stem}.episodes.csv")
    with episode_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = tuple(
            key
            for key in report["episodes"][0]
            if key != "reward_components"
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in record.items()
                if key != "reward_components"
            }
            for record in report["episodes"]
        )


def _evaluate_runner_v2_unleased(
    checkpoint: Path,
    output: Path,
    *,
    episodes_per_tier: int = 500,
    tiers: str | Iterable[int] = ALL_TIERS,
    directives: str | Iterable[ContractDirective | str | int] = "standard",
    seed_start: int = FINAL_TEST_SEED_START,
    workers: int = 0,
    security_checkpoints: Sequence[Path] = (),
    security_pool_salt: int = 0,
    overwrite: bool = False,
) -> Path:
    """Evaluate one immutable v2 checkpoint on an explicit final-test slice."""

    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"v2 runner checkpoint is missing: {checkpoint}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists; choose a new slice/output or opt into overwrite"
        )
    if episodes_per_tier <= 0:
        raise ValueError("episodes_per_tier must be positive")
    if seed_start < FINAL_TEST_SEED_START:
        raise ValueError(
            f"seed_start must be at least {FINAL_TEST_SEED_START:,}"
        )
    selected_tiers = _parse_tiers(tiers)
    selected_directives = _parse_directives(directives)
    security_paths = tuple(
        str(Path(path).expanduser().resolve())
        for path in security_checkpoints
    )
    for path in security_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"security checkpoint is missing: {path}")
    # Fail closed before dispatching worker processes.
    load_runner_v2(checkpoint, device="cpu")
    tasks = [
        (
            tier,
            int(directive),
            episode,
            final_test_seed(seed_start, tier, episode),
        )
        for directive in selected_directives
        for tier in selected_tiers
        for episode in range(episodes_per_tier)
    ]
    worker_count = (
        max(1, min(len(tasks), (os.cpu_count() or 2) - 1, 16))
        if workers == 0
        else int(workers)
    )
    if worker_count <= 0:
        raise ValueError("workers must be non-negative")
    if worker_count == 1:
        _init_worker(str(checkpoint), security_paths, security_pool_salt)
        records = [_episode(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(
                str(checkpoint),
                security_paths,
                security_pool_salt,
            ),
        ) as executor:
            records = list(executor.map(_episode, tasks, chunksize=1))
    if [
        (record["tier"], record["directive"], record["episode_index"], record["seed"])
        for record in records
    ] != [
        (
            tier,
            ContractDirective(directive).name.lower(),
            episode,
            seed,
        )
        for tier, directive, episode, seed in tasks
    ]:
        raise RuntimeError("parallel v2 evaluation changed deterministic task order")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        directive.name.lower(): {
            str(tier): [] for tier in selected_tiers
        }
        for directive in selected_directives
    }
    for record in records:
        grouped[record["directive"]][str(record["tier"])].append(record)
    directive_summaries = {
        directive: {
            tier: _summary(rows)
            for tier, rows in tier_rows.items()
        }
        for directive, tier_rows in grouped.items()
    }
    tier_gate = {
        str(tier): min(
            float(directive_summaries[directive.name.lower()][str(tier)]["success_rate"])
            for directive in selected_directives
        )
        for tier in selected_tiers
    }
    acceptance = (
        set(selected_tiers) == set(ALL_TIERS)
        and all(
            tier_gate[str(tier)] >= ACCEPTANCE_THRESHOLDS[tier]
            for tier in ALL_TIERS
        )
    )
    report = {
        "contract": EVALUATION_CONTRACT_V2,
        "observation_contract": OBSERVATION_CONTRACT_V2,
        "environment_fingerprint": multi_agent_environment_fingerprint(),
        "model_fingerprint": runner_model_fingerprint(),
        "runner_checkpoint": str(checkpoint),
        "runner_checkpoint_sha256": _sha256(checkpoint),
        "security_checkpoints": [
            {
                "path": path,
                "sha256": _sha256(Path(path)),
            }
            for path in security_paths
        ],
        "seed_start": seed_start,
        "seed_namespace": f"final-test-{seed_start}",
        "episodes_per_tier_per_directive": episodes_per_tier,
        "tiers": selected_tiers,
        "evaluated_directives": tuple(
            directive.name.lower() for directive in selected_directives
        ),
        "workers": worker_count,
        "tier_worst_directive_success": tier_gate,
        "acceptance_passed": acceptance,
        "meets_acceptance_thresholds": acceptance,
        "directives": directive_summaries,
        "episodes": records,
    }
    _write_reports(output, report)
    return output


def evaluate_runner_v2_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    episodes_per_tier: int = 500,
    tiers: str | Iterable[int] = ALL_TIERS,
    directives: str | Iterable[ContractDirective | str | int] = "standard",
    seed_start: int = FINAL_TEST_SEED_START,
    workers: int = 0,
    security_checkpoints: Sequence[Path] = (),
    security_pool_salt: int = 0,
    slice_manifest: Path = DEFAULT_V2_SLICE_MANIFEST,
    overwrite: bool = False,
) -> Path:
    """Consume one explicitly reserved v2 final-test slice exactly once."""

    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    if overwrite:
        raise ValueError(
            "final-test reports are immutable; reserve a new slice and output"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"v2 runner checkpoint is missing: {checkpoint}"
        )
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; reserve a new slice and output"
        )
    if episodes_per_tier <= 0:
        raise ValueError("episodes_per_tier must be positive")
    if seed_start < FINAL_TEST_SEED_START:
        raise ValueError(
            f"seed_start must be at least {FINAL_TEST_SEED_START:,}"
        )
    selected_tiers = _parse_tiers(tiers)
    selected_directives = _parse_directives(directives)
    for path in security_checkpoints:
        if not Path(path).expanduser().resolve().is_file():
            raise FileNotFoundError(
                f"security checkpoint is missing: {path}"
            )
    # Validate checkpoint semantics before opening the one-way slice.
    load_runner_v2(checkpoint, device="cpu")
    policy_kind = "runner-v2-neural:" + ",".join(
        directive.name.lower() for directive in selected_directives
    )
    from ghostline.evaluation import _open_final_slice

    lease = _open_final_slice(
        manifest_path=Path(slice_manifest),
        seed_start=seed_start,
        episodes=episodes_per_tier,
        tiers=selected_tiers,
        environment_fingerprint=multi_agent_environment_fingerprint(),
        policy_kind=policy_kind,
        checkpoint_sha256=_sha256(checkpoint),
        output=output,
    )
    try:
        result = _evaluate_runner_v2_unleased(
            checkpoint,
            output,
            episodes_per_tier=episodes_per_tier,
            tiers=selected_tiers,
            directives=selected_directives,
            seed_start=seed_start,
            workers=workers,
            security_checkpoints=security_checkpoints,
            security_pool_salt=security_pool_salt,
            overwrite=False,
        )
        report = json.loads(result.read_text(encoding="utf-8"))
        outputs = (
            result,
            result.with_suffix(".csv"),
            result.with_name(f"{result.stem}.episodes.csv"),
        )
        lease.finalize(report, outputs)
        return result
    except BaseException as error:
        lease.abort(error)
        raise
