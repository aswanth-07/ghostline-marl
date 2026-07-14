from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import fmean, median
import time
from typing import Any, Iterable, Sequence

import numpy as np

from ghostline.env import GhostlineEnv
from ghostline.model import current_environment_fingerprint, load_policy
from ghostline.policies import FairScriptedPolicy, ObservationTeacherPolicy
from ghostline.seeds import FINAL_TEST_SEED_START, final_test_seed
from ghostline.simulation import GhostlineSimulation
from ghostline.types import Action


DEFAULT_RELEASE_SEED_START = 8_000_000
DEFAULT_SLICE_MANIFEST = Path("benchmarks/final-test-slices.json")
FINAL_REPORT_CONTRACT = "ghostline-final-evaluation-v2"
FINAL_SLICE_MANIFEST_CONTRACT = "ghostline-final-test-slices-v1"
OBSERVATION_CONTRACT = "GhostlineEnv-v2"
ALL_TIERS = tuple(range(1, 7))
ACCEPTANCE = {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.85}


_WORKER_POLICY = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _action_sequence_hash(actions: Iterable[int]) -> str:
    """Hash the exact flattened action sequence without text-format ambiguity."""

    digest = hashlib.sha256()
    for action in actions:
        value = int(action)
        if not 0 <= value < 36:
            raise ValueError(f"action must lie in 0..35, got {value}")
        digest.update(bytes((value,)))
    return digest.hexdigest()


def _episode_record(
    info: dict[str, Any],
    *,
    tier: int,
    seed: int,
    episode_index: int,
    actions: Sequence[int],
    policy_latencies_ms: Sequence[float] = (),
) -> dict[str, Any]:
    telemetry = info.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    reward_components = info.get("reward_components")
    if not isinstance(reward_components, dict):
        reward_components = {
            key.removeprefix("reward_"): value
            for key, value in info.items()
            if key.startswith("reward_")
            and key != "reward_total"
            and isinstance(value, (int, float))
        }
    normalized_rewards = {
        str(key): float(value)
        for key, value in sorted(reward_components.items())
        if isinstance(value, (int, float))
    }
    computed_reward_total = float(sum(normalized_rewards.values()))
    declared_reward_total = info.get("reward_total", computed_reward_total)
    if not isinstance(declared_reward_total, (int, float)) or not math.isclose(
        float(declared_reward_total), computed_reward_total, abs_tol=1e-9
    ):
        raise RuntimeError("terminal reward components do not sum to reward_total")
    return {
        "tier": int(tier),
        "episode_index": int(episode_index),
        "seed": int(seed),
        "success": bool(info["is_success"]),
        "failure_reason": str(info["fail_reason"]),
        "duration_seconds": float(info["duration_seconds"]),
        "max_trace": float(info["max_trace"]),
        "damage": int(info["damage"]),
        "damage_by_guard": int(info.get("damage_by_guard", 0)),
        "damage_by_drone": int(info.get("damage_by_drone", 0)),
        "detections": int(info["detections"]),
        "optional_data": int(info["optional_data"]),
        "pulse_uses": int(info.get("pulse_uses", 0)),
        "path_efficiency": (
            float(telemetry["path_efficiency"])
            if telemetry.get("path_efficiency") is not None
            else None
        ),
        "decision_count": len(actions),
        "action_sha256": _action_sequence_hash(actions),
        "policy_latency_total_ms": float(sum(policy_latencies_ms)),
        "median_policy_latency_ms": (
            float(median(policy_latencies_ms)) if policy_latencies_ms else None
        ),
        "reward_total": computed_reward_total,
        "reward_components": normalized_rewards,
    }


def _init_model_worker(model: str) -> None:
    global _WORKER_POLICY
    import torch

    torch.set_num_threads(1)
    _WORKER_POLICY = load_policy(Path(model))


def _model_episode(args: tuple[int, int, int]) -> dict[str, Any]:
    tier, seed, episode_index = args
    if _WORKER_POLICY is None:
        raise RuntimeError("model evaluation worker was not initialized")
    env = GhostlineEnv(seed=seed, tier=tier)
    observation, _ = env.reset(seed=seed)
    hidden = None
    actions: list[int] = []
    latencies: list[float] = []
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            started = time.perf_counter()
            action, hidden = _WORKER_POLICY.act(observation, hidden, deterministic=True)
            latencies.append((time.perf_counter() - started) * 1000.0)
            actions.append(int(action))
            observation, _, terminated, truncated, info = env.step(action)
        return _episode_record(
            dict(info),
            tier=tier,
            seed=seed,
            episode_index=episode_index,
            actions=actions,
            policy_latencies_ms=latencies,
        )
    finally:
        env.close()


def _baseline_episode(args: tuple[int, int, int, str]) -> dict[str, Any]:
    tier, seed, episode_index, baseline = args
    actions: list[int] = []
    if baseline == "legacy_scripted":
        sim = GhostlineSimulation(seed=seed, tier=tier)
        policy = FairScriptedPolicy()
        while not (sim.terminated or sim.truncated):
            action = int(policy.act(sim))
            actions.append(action)
            sim.advance(Action.decode(action), ticks=6)
        return _episode_record(
            sim.terminal_info(),
            tier=tier,
            seed=seed,
            episode_index=episode_index,
            actions=actions,
        )

    env = GhostlineEnv(seed=seed, tier=tier)
    observation, _ = env.reset(seed=seed)
    teacher = ObservationTeacherPolicy()
    rng = np.random.default_rng(seed + 73)
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            if baseline == "teacher":
                action = int(teacher.act(observation))
            elif baseline == "random":
                legal = np.flatnonzero(observation["action_mask"])
                action = int(rng.choice(legal))
            else:
                raise ValueError(f"Unknown baseline: {baseline}")
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
        return _episode_record(
            dict(info),
            tier=tier,
            seed=seed,
            episode_index=episode_index,
            actions=actions,
        )
    finally:
        env.close()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return float(fmean(retained)) if retained else None


def _median_optional(values: Iterable[float | None]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return float(median(retained)) if retained else None


def _aggregate_tier(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty tier")
    successes = sum(int(row["success"]) for row in rows)
    failures = Counter(
        str(row["failure_reason"])
        for row in rows
        if not bool(row["success"])
    )
    decision_count = sum(int(row["decision_count"]) for row in rows)
    latency_total = sum(float(row["policy_latency_total_ms"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95": list(wilson(successes, len(rows))),
        "median_duration_seconds": float(median(float(row["duration_seconds"]) for row in rows)),
        "median_max_trace": float(median(float(row["max_trace"]) for row in rows)),
        "mean_damage": float(fmean(float(row["damage"]) for row in rows)),
        "mean_guard_damage": float(fmean(float(row["damage_by_guard"]) for row in rows)),
        "mean_drone_damage": float(fmean(float(row["damage_by_drone"]) for row in rows)),
        "mean_detections": float(fmean(float(row["detections"]) for row in rows)),
        "optional_data_rate": float(fmean(float(int(row["optional_data"]) > 0) for row in rows)),
        "mean_path_efficiency": _mean_optional(row["path_efficiency"] for row in rows),
        "mean_policy_latency_ms": latency_total / decision_count if decision_count else None,
        "median_episode_policy_latency_ms": _median_optional(
            row["median_policy_latency_ms"] for row in rows
        ),
        "failure_reasons": {key: failures[key] for key in sorted(failures)},
    }


def _build_report(
    *,
    model: Path | None,
    checkpoint_sha256: str | None,
    baseline: str,
    seed_start: int,
    episodes: int,
    tiers: Sequence[int],
    environment_fingerprint: str,
    audit_id: str,
    slice_manifest: Path | None,
    episode_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        (dict(row) for row in episode_records),
        key=lambda row: (int(row["tier"]), int(row["episode_index"])),
    )
    expected = [
        (tier, episode, final_test_seed(seed_start, tier, episode))
        for tier in tiers
        for episode in range(episodes)
    ]
    actual = [
        (int(row["tier"]), int(row["episode_index"]), int(row["seed"]))
        for row in ordered
    ]
    if actual != expected:
        raise RuntimeError("final evaluation results do not exactly match the scheduled seed order")

    tier_reports = {
        str(tier): _aggregate_tier([row for row in ordered if int(row["tier"]) == tier])
        for tier in tiers
    }
    thresholds = {str(tier): ACCEPTANCE[tier] for tier in tiers}
    meets = all(
        float(tier_reports[str(tier)]["success_rate"]) >= ACCEPTANCE[tier]
        for tier in tiers
    )
    return {
        "report_contract": FINAL_REPORT_CONTRACT,
        "observation_contract": OBSERVATION_CONTRACT,
        "environment_fingerprint": environment_fingerprint,
        "deterministic_actions": True,
        "policy_kind": "neural" if model is not None else f"baseline:{baseline}",
        "policy": model.as_posix() if model is not None else baseline,
        "checkpoint_sha256": checkpoint_sha256,
        "release_audit": slice_manifest is not None,
        "seed_namespace": f"final_test_{seed_start:_}_plus",
        "seed_start": int(seed_start),
        "seed_formula": "seed_start + tier * 100000 + episode_index",
        "episodes_per_tier": int(episodes),
        "tiers_evaluated": list(tiers),
        "acceptance_thresholds": thresholds,
        "meets_acceptance_thresholds": meets,
        "slice_manifest": slice_manifest.as_posix() if slice_manifest is not None else None,
        "slice_audit_id": audit_id,
        "tiers": tier_reports,
        "episode_records": ordered,
    }


def _aggregate_csv(report: dict[str, Any]) -> str:
    fields = (
        "environment_fingerprint",
        "checkpoint_sha256",
        "seed_start",
        "tier",
        "episodes",
        "successes",
        "success_rate",
        "wilson_low",
        "wilson_high",
        "median_duration_seconds",
        "median_max_trace",
        "mean_damage",
        "mean_guard_damage",
        "mean_drone_damage",
        "mean_detections",
        "optional_data_rate",
        "mean_path_efficiency",
        "mean_policy_latency_ms",
        "failure_reasons_json",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for tier in report["tiers_evaluated"]:
        row = report["tiers"][str(tier)]
        writer.writerow(
            {
                "environment_fingerprint": report["environment_fingerprint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "seed_start": report["seed_start"],
                "tier": tier,
                "episodes": row["episodes"],
                "successes": row["successes"],
                "success_rate": row["success_rate"],
                "wilson_low": row["wilson_95"][0],
                "wilson_high": row["wilson_95"][1],
                "median_duration_seconds": row["median_duration_seconds"],
                "median_max_trace": row["median_max_trace"],
                "mean_damage": row["mean_damage"],
                "mean_guard_damage": row["mean_guard_damage"],
                "mean_drone_damage": row["mean_drone_damage"],
                "mean_detections": row["mean_detections"],
                "optional_data_rate": row["optional_data_rate"],
                "mean_path_efficiency": row["mean_path_efficiency"],
                "mean_policy_latency_ms": row["mean_policy_latency_ms"],
                "failure_reasons_json": json.dumps(row["failure_reasons"], sort_keys=True),
            }
        )
    return output.getvalue()


def _episodes_csv(report: dict[str, Any]) -> str:
    fields = (
        "environment_fingerprint",
        "checkpoint_sha256",
        "seed_start",
        "tier",
        "episode_index",
        "seed",
        "success",
        "failure_reason",
        "duration_seconds",
        "max_trace",
        "damage",
        "damage_by_guard",
        "damage_by_drone",
        "detections",
        "optional_data",
        "pulse_uses",
        "path_efficiency",
        "decision_count",
        "action_sha256",
        "policy_latency_total_ms",
        "median_policy_latency_ms",
        "reward_total",
        "reward_components_json",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["episode_records"]:
        writer.writerow(
            {
                "environment_fingerprint": report["environment_fingerprint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "seed_start": report["seed_start"],
                **{key: row[key] for key in fields if key in row},
                "reward_components_json": json.dumps(row["reward_components"], sort_keys=True),
            }
        )
    return output.getvalue()


def _load_slice_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read final-test slice manifest: {path}") from error
    if manifest.get("manifest_contract") != FINAL_SLICE_MANIFEST_CONTRACT:
        raise RuntimeError("final-test slice manifest has an unsupported contract")
    if manifest.get("observation_contract") != OBSERVATION_CONTRACT:
        raise RuntimeError("final-test slice manifest does not declare GhostlineEnv-v2")
    if not isinstance(manifest.get("slices"), list):
        raise RuntimeError("final-test slice manifest must contain a slices list")
    return manifest


def _slice_entry(manifest: dict[str, Any], seed_start: int) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["slices"]
        if isinstance(item, dict) and int(item.get("seed_start", -1)) == int(seed_start)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"final-test seed_start {seed_start:,} must have exactly one explicit manifest entry"
        )
    return matches[0]


def _audit_id(
    *,
    seed_start: int,
    episodes: int,
    tiers: Sequence[int],
    environment_fingerprint: str,
    policy_kind: str,
    checkpoint_sha256: str | None,
) -> str:
    payload = {
        "checkpoint_sha256": checkpoint_sha256,
        "environment_fingerprint": environment_fingerprint,
        "episodes_per_tier": episodes,
        "policy_kind": policy_kind,
        "seed_start": seed_start,
        "tiers": list(tiers),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class FinalSliceLease:
    manifest_path: Path
    lock_path: Path
    seed_start: int
    audit_id: str

    def _entry_for_active_audit(self, manifest: dict[str, Any]) -> dict[str, Any]:
        entry = _slice_entry(manifest, self.seed_start)
        active = entry.get("active_audit")
        if entry.get("status") != "opened_locked" or not isinstance(active, dict):
            raise RuntimeError("final-test slice is no longer held by an active audit")
        if active.get("audit_id") != self.audit_id:
            raise RuntimeError("final-test slice audit identity changed while it was open")
        return entry

    def finalize(self, report: dict[str, Any], outputs: Sequence[Path]) -> None:
        manifest = _load_slice_manifest(self.manifest_path)
        entry = self._entry_for_active_audit(manifest)
        entry["status"] = "consumed"
        entry["result"] = {
            "audit_id": self.audit_id,
            "meets_acceptance_thresholds": bool(report["meets_acceptance_thresholds"]),
            "outputs": [
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in outputs
            ],
        }
        entry.pop("active_audit", None)
        _atomic_write(self.manifest_path, _stable_json(manifest))
        self.lock_path.unlink(missing_ok=False)

    def abort(self, error: BaseException) -> None:
        manifest = _load_slice_manifest(self.manifest_path)
        entry = self._entry_for_active_audit(manifest)
        entry["status"] = "aborted_retired"
        entry["result"] = {
            "audit_id": self.audit_id,
            "error_type": type(error).__name__,
            "note": "The slice remains retired because an evaluation attempt began.",
        }
        entry.pop("active_audit", None)
        _atomic_write(self.manifest_path, _stable_json(manifest))
        self.lock_path.unlink(missing_ok=False)


def _open_final_slice(
    *,
    manifest_path: Path,
    seed_start: int,
    episodes: int,
    tiers: Sequence[int],
    environment_fingerprint: str,
    policy_kind: str,
    checkpoint_sha256: str | None,
    output: Path,
) -> FinalSliceLease:
    manifest_path = Path(manifest_path)
    lock_path = manifest_path.with_name(f"{manifest_path.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"final-test slice manifest is already locked: {lock_path}; "
            "a crashed/open audit must be retired manually, never reopened"
        ) from error
    os.close(descriptor)
    try:
        manifest = _load_slice_manifest(manifest_path)
        if manifest.get("environment_fingerprint") != environment_fingerprint:
            raise RuntimeError("final-test slice manifest fingerprint does not match the frozen environment")
        entry = _slice_entry(manifest, seed_start)
        if entry.get("status") != "reserved_unopened":
            raise RuntimeError(
                f"final-test slice {seed_start:,} has status {entry.get('status')!r}; "
                "only reserved_unopened slices may be opened"
            )
        if entry.get("environment_fingerprint") != environment_fingerprint:
            raise RuntimeError("reserved final-test slice fingerprint does not match the frozen environment")
        if entry.get("policy_kind") != policy_kind:
            raise RuntimeError(
                f"reserved final-test slice expects policy_kind={entry.get('policy_kind')!r}, "
                f"not {policy_kind!r}"
            )
        if int(entry.get("episodes_per_tier", -1)) != episodes:
            raise RuntimeError("final-test slice requires its declared episodes_per_tier")
        if [int(value) for value in entry.get("tiers", [])] != list(tiers):
            raise RuntimeError("final-test slice requires its declared complete tier set")
        audit_id = _audit_id(
            seed_start=seed_start,
            episodes=episodes,
            tiers=tiers,
            environment_fingerprint=environment_fingerprint,
            policy_kind=policy_kind,
            checkpoint_sha256=checkpoint_sha256,
        )
        entry["status"] = "opened_locked"
        entry["active_audit"] = {
            "audit_id": audit_id,
            "checkpoint_sha256": checkpoint_sha256,
            "environment_fingerprint": environment_fingerprint,
            "episodes_per_tier": episodes,
            "output": output.as_posix(),
            "policy_kind": policy_kind,
            "tiers": list(tiers),
        }
        _atomic_write(manifest_path, _stable_json(manifest))
        return FinalSliceLease(manifest_path, lock_path, seed_start, audit_id)
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise


def evaluate(
    *,
    model: Path | None,
    episodes: int,
    tier: int | None,
    output: Path,
    baseline: str = "teacher",
    workers: int = 0,
    seed_start: int = DEFAULT_RELEASE_SEED_START,
    slice_manifest: Path | None = DEFAULT_SLICE_MANIFEST,
) -> dict[str, Any]:
    """Consume one explicitly reserved final-test slice exactly once.

    This is a release-audit entry point, not a checkpoint-selection utility.
    Candidate selection must use ``scripts/evaluate_validation_policy.py``.
    """

    if model is None and baseline not in ("teacher", "random", "legacy_scripted"):
        raise ValueError("baseline must be teacher, random, or legacy_scripted")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative; zero selects an automatic count")
    if seed_start < FINAL_TEST_SEED_START:
        raise ValueError(f"final evaluation seed_start must be at least {FINAL_TEST_SEED_START:,}")
    output = Path(output)
    if output.suffix.casefold() != ".json":
        raise ValueError("final evaluation output must use a .json suffix")
    tiers = (tier,) if tier is not None else ALL_TIERS
    model_path = Path(model) if model is not None else None
    policy_kind = "neural" if model_path is not None else f"baseline:{baseline}"
    environment_fingerprint = current_environment_fingerprint()
    checkpoint_sha256: str | None = None

    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"policy checkpoint does not exist: {model_path}")
        checkpoint_sha256 = _sha256(model_path)
        # Fingerprint/architecture preflight occurs before the one-way slice opens.
        policy = load_policy(model_path, device="cpu")
        del policy

    output_paths = (output, output.with_suffix(".csv"), output.with_suffix(".episodes.csv"))
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "final evaluation never overwrites evidence: " + ", ".join(path.as_posix() for path in existing)
        )

    if slice_manifest is None:
        lease = None
        audit_id = _audit_id(
            seed_start=seed_start,
            episodes=episodes,
            tiers=tiers,
            environment_fingerprint=environment_fingerprint,
            policy_kind=policy_kind,
            checkpoint_sha256=checkpoint_sha256,
        )
    else:
        lease = _open_final_slice(
            manifest_path=Path(slice_manifest),
            seed_start=seed_start,
            episodes=episodes,
            tiers=tiers,
            environment_fingerprint=environment_fingerprint,
            policy_kind=policy_kind,
            checkpoint_sha256=checkpoint_sha256,
            output=output,
        )
        audit_id = lease.audit_id
    try:
        if model_path is None:
            jobs = [
                (tier_number, final_test_seed(seed_start, tier_number, episode), episode, baseline)
                for tier_number in tiers
                for episode in range(episodes)
            ]
            worker_count = max(1, min(12, len(jobs), (os.cpu_count() or 4) - 2))
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                episode_records = list(pool.map(_baseline_episode, jobs))
        else:
            jobs = [
                (tier_number, final_test_seed(seed_start, tier_number, episode), episode)
                for tier_number in tiers
                for episode in range(episodes)
            ]
            worker_count = workers or max(1, min(6, len(jobs), (os.cpu_count() or 4) // 3))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_model_worker,
                initargs=(str(model_path),),
            ) as pool:
                episode_records = list(pool.map(_model_episode, jobs))

        if current_environment_fingerprint() != environment_fingerprint:
            raise RuntimeError(
                "environment fingerprint changed during final evaluation; the opened slice is retired"
            )
        if model_path is not None and _sha256(model_path) != checkpoint_sha256:
            raise RuntimeError(
                "policy checkpoint changed during final evaluation; the opened slice is retired"
            )

        report = _build_report(
            model=model_path,
            checkpoint_sha256=checkpoint_sha256,
            baseline=baseline,
            seed_start=seed_start,
            episodes=episodes,
            tiers=tiers,
            environment_fingerprint=environment_fingerprint,
            audit_id=audit_id,
            slice_manifest=Path(slice_manifest) if slice_manifest is not None else None,
            episode_records=episode_records,
        )
        _atomic_write(output.with_suffix(".episodes.csv"), _episodes_csv(report))
        _atomic_write(output.with_suffix(".csv"), _aggregate_csv(report))
        _atomic_write(output, _stable_json(report))
        if lease is not None:
            lease.finalize(report, output_paths)
    except BaseException as error:
        if lease is not None:
            try:
                lease.abort(error)
            except BaseException as audit_error:
                raise RuntimeError(
                    "final evaluation failed and its slice lock could not be finalized; "
                    f"leave {lease.lock_path} in place and retire the slice manually"
                ) from audit_error
        raise
    print(_stable_json(report), end="")
    return report
