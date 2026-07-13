"""Fair teacher trajectory collection, behavior cloning, and DAgger recovery training."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ghostline.env import GhostlineEnv
from ghostline.model import (
    OBSERVATION_CONTRACT,
    UniversalGhostlinePolicy,
    load_policy,
    require_current_checkpoint,
    save_policy,
)
from ghostline.policies import ObservationTeacherPolicy


OBS_KEYS = (
    "ego",
    "objective",
    "local_grid",
    "targets",
    "target_mask",
    "entities",
    "entity_mask",
    "rays",
    "action_mask",
)

TRAINING_SEED_LIMIT = 1_000_000
DAGGER_ROUND_SEED_STRIDE = 10_000
DEFAULT_DAGGER_COLLECTION_WORKERS = 12
DEFAULT_BC_SEQUENCE_LENGTH = 64
DEFAULT_BC_BURN_IN = 32
DEFAULT_VALIDATION_WINDOWS = 128
DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_UNIFORM_WINDOW_FRACTION = 0.50
WINDOW_STRATA = (
    "uniform",
    "endpoint",
    "action_change",
    "dash",
    "pulse",
    "recovery",
)
WINDOW_SAMPLER_VERSION = "ghostline-window-strata-v1"
SPECIALIZED_WINDOW_WEIGHTS = {
    "endpoint": 0.20,
    "action_change": 0.30,
    "dash": 0.10,
    "pulse": 0.20,
    "recovery": 0.20,
}
SPECIALIZED_TIE_ORDER = ("endpoint", "pulse", "recovery", "dash", "action_change")
ANCHOR_LOSS_WEIGHTS = {
    "uniform": 1.0,
    "endpoint": 2.0,
    "action_change": 2.0,
    "dash": 2.0,
    "pulse": 6.0,
    "recovery": 4.0,
}
_EPISODE_FILE_PATTERN = re.compile(r"tier-(\d+)-seed-(\d+)\.npz")


_COLLECTION_BEHAVIOR: UniversalGhostlinePolicy | None = None
_COLLECTION_DEVICE = "cpu"


def training_environment_fingerprint() -> str:
    """Hash the simulator/controller contract used to produce imitation data."""
    package = Path(__file__).resolve().parent
    names = ("config.py", "env.py", "generation.py", "policies.py", "simulation.py", "types.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CollectionSummary:
    episodes: int
    transitions: int
    behavior_successes: int
    output: Path
    attempted_episodes: int = 0
    discarded_episodes: int = 0

    @property
    def behavior_success_rate(self) -> float:
        return self.behavior_successes / max(1, self.episodes)


@dataclass(frozen=True)
class _PureTeacherJob:
    output: Path
    requested_tier: int
    seed: int
    lesson: int
    success_only: bool


@dataclass(frozen=True)
class _BehaviorJob:
    output: Path
    requested_tier: int
    seed: int
    lesson: int
    teacher_probability: float


@dataclass(frozen=True)
class _EpisodeResult:
    requested_tier: int
    tier: int
    seed: int
    transitions: int
    behavior_success: int
    fail_reason: str
    path: Path | None


def _auxiliary_labels(observation: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.float32]:
    objective = np.asarray(observation["objective"], dtype=np.float32)
    direction = objective[4:6] if float(np.linalg.norm(objective[4:6])) > 0.04 else objective[1:3]
    magnitude = float(np.linalg.norm(direction))
    bearing = direction / magnitude if magnitude > 1e-6 else np.zeros(2, dtype=np.float32)
    danger = np.float32(np.max(np.asarray(observation["rays"], dtype=np.float32)[:, 1], initial=0.0))
    return bearing.astype(np.float32), danger


def _discounted_returns(rewards: list[float], gamma: float = 0.995) -> np.ndarray:
    output = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = float(rewards[index]) + gamma * running
        output[index] = running
    return output


def factorized_action_nll(
    masked_logits: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Marginal move/dash/pulse NLLs from the legal 36-way joint policy.

    The public action ordering is ``move + 9 * dash + 18 * pulse``, so the
    final dimension reshapes to ``[pulse=2, dash=2, move=9]``. Log-marginals
    are derived from the already action-masked joint distribution; no new
    policy heads or inference contract are introduced.
    """
    if masked_logits.shape[-1] != 36:
        raise ValueError("factorized action loss requires 36 joint-action logits")
    if masked_logits.shape[:-1] != actions.shape:
        raise ValueError("action labels must match the leading logit dimensions")
    joint_log_probability = torch.log_softmax(masked_logits, dim=-1).reshape(
        *masked_logits.shape[:-1], 2, 2, 9
    )
    move_log_probability = torch.logsumexp(joint_log_probability, dim=(-3, -2))
    dash_log_probability = torch.logsumexp(joint_log_probability, dim=(-3, -1))
    pulse_log_probability = torch.logsumexp(joint_log_probability, dim=(-2, -1))
    action_values = actions.long()
    move = action_values.remainder(9)
    dash = action_values.div(9, rounding_mode="floor").remainder(2)
    pulse = action_values.div(18, rounding_mode="floor")

    def gather_nll(log_probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -log_probability.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    return (
        gather_nll(move_log_probability, move),
        gather_nll(dash_log_probability, dash),
        gather_nll(pulse_log_probability, pulse),
    )


def _write_episode(
    output: Path,
    *,
    tier: int,
    requested_tier: int,
    seed: int,
    observations: list[dict[str, np.ndarray]],
    actions: list[int],
    behavior_actions: list[int],
    rewards: list[float],
    bearings: list[np.ndarray],
    dangers: list[np.float32],
) -> Path:
    payload: dict[str, np.ndarray] = {
        key: np.stack([observation[key] for observation in observations]) for key in OBS_KEYS
    }
    payload.update(
        action=np.asarray(actions, dtype=np.int64),
        behavior_action=np.asarray(behavior_actions, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float32),
        return_= _discounted_returns(rewards),
        objective_bearing=np.asarray(bearings, dtype=np.float32),
        danger=np.asarray(dangers, dtype=np.float32),
        tier=np.asarray(tier, dtype=np.int8),
        requested_tier=np.asarray(requested_tier, dtype=np.int8),
        seed=np.asarray(seed, dtype=np.int64),
    )
    path = output / f"tier-{tier}-seed-{seed}.npz"
    np.savez_compressed(path, **payload)
    return path


def _collect_episode(
    output: Path,
    *,
    requested_tier: int,
    seed: int,
    lesson: int,
    behavior: UniversalGhostlinePolicy | None = None,
    teacher_probability: float = 1.0,
    rng: np.random.Generator | None = None,
    device: str | torch.device = "cpu",
    success_only: bool = False,
) -> _EpisodeResult:
    """Roll out one episode and write it directly when it should be retained."""
    env = GhostlineEnv(seed=seed, tier=requested_tier)
    try:
        options = {"tier": requested_tier}
        if lesson:
            options["training_lesson"] = lesson
        observation, reset_info = env.reset(seed=seed, options=options)
        actual_tier = int(reset_info.get("tier", env.tier))
        teacher = ObservationTeacherPolicy()
        hidden = None
        observations: list[dict[str, np.ndarray]] = []
        labels: list[int] = []
        executed: list[int] = []
        rewards: list[float] = []
        bearings: list[np.ndarray] = []
        dangers: list[np.float32] = []
        terminated = truncated = False
        info: dict[str, object] = reset_info
        while not (terminated or truncated):
            teacher_action = teacher.act(observation)
            behavior_action = teacher_action
            if behavior is not None:
                policy_action, next_hidden = behavior.act(
                    observation, hidden, deterministic=True, device=device
                )
                hidden = next_hidden
                if rng is None:
                    raise RuntimeError("A seeded RNG is required for behavior-policy collection")
                if rng.random() >= teacher_probability:
                    behavior_action = policy_action
            teacher.observe_executed_action(behavior_action)
            bearing, danger = _auxiliary_labels(observation)
            observations.append({key: np.asarray(observation[key]).copy() for key in OBS_KEYS})
            labels.append(teacher_action)
            executed.append(behavior_action)
            bearings.append(bearing)
            dangers.append(danger)
            observation, reward, terminated, truncated, info = env.step(behavior_action)
            rewards.append(float(reward))

        success = int(bool(info.get("is_success", False)))
        path: Path | None = None
        if success or not success_only:
            path = _write_episode(
                output,
                tier=actual_tier,
                requested_tier=requested_tier,
                seed=seed,
                observations=observations,
                actions=labels,
                behavior_actions=executed,
                rewards=rewards,
                bearings=bearings,
                dangers=dangers,
            )
        else:
            # Do not allow a stale file from an earlier non-filtered collection
            # to masquerade as a retained successful trajectory.
            (output / f"tier-{actual_tier}-seed-{seed}.npz").unlink(missing_ok=True)
        return _EpisodeResult(
            requested_tier=requested_tier,
            tier=actual_tier,
            seed=seed,
            transitions=len(labels),
            behavior_success=success,
            fail_reason=str(info.get("fail_reason", "unknown")),
            path=path,
        )
    finally:
        env.close()


def _collect_pure_teacher_episode(job: _PureTeacherJob) -> _EpisodeResult:
    """Pickle-safe process worker for observation-only teacher collection."""
    return _collect_episode(
        job.output,
        requested_tier=job.requested_tier,
        seed=job.seed,
        lesson=job.lesson,
        success_only=job.success_only,
    )


def _episode_mixture_rng(
    *, requested_tier: int, seed: int, lesson: int
) -> np.random.Generator:
    """Return an episode-local DAgger RNG independent of scheduling order."""
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(requested_tier), int(lesson), 0xDA66E2]
    )
    return np.random.default_rng(seed_sequence)


def _init_behavior_worker(checkpoint: str, collection_device: str) -> None:
    """Load one behavior policy per persistent process worker."""
    global _COLLECTION_BEHAVIOR, _COLLECTION_DEVICE
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work begins. A
        # spawned worker normally takes the first branch, but thread count for
        # model kernels remains authoritatively pinned above either way.
        pass
    _COLLECTION_DEVICE = str(torch.device(collection_device))
    _COLLECTION_BEHAVIOR = load_policy(Path(checkpoint), device=_COLLECTION_DEVICE)


def _collect_behavior_episode(job: _BehaviorJob) -> _EpisodeResult:
    """Pickle-safe worker using the process-local behavior checkpoint."""
    if _COLLECTION_BEHAVIOR is None:
        raise RuntimeError("DAgger behavior worker was not initialized")
    return _collect_episode(
        job.output,
        requested_tier=job.requested_tier,
        seed=job.seed,
        lesson=job.lesson,
        behavior=_COLLECTION_BEHAVIOR,
        teacher_probability=job.teacher_probability,
        rng=_episode_mixture_rng(
            requested_tier=job.requested_tier,
            seed=job.seed,
            lesson=job.lesson,
        ),
        device=_COLLECTION_DEVICE,
    )


def _record(result: _EpisodeResult) -> dict[str, int | str]:
    return {
        "requested_tier": result.requested_tier,
        "tier": result.tier,
        "seed": result.seed,
        "transitions": result.transitions,
        "behavior_success": result.behavior_success,
        "fail_reason": result.fail_reason,
    }


def collect_trajectories(
    output: Path,
    *,
    tiers: Iterable[int],
    episodes_per_tier: int,
    seed_start: int = 0,
    lesson: int = 0,
    behavior_checkpoint: Path | None = None,
    teacher_probability: float = 1.0,
    recurrent_size: int = 384,
    collection_device: str | torch.device = "cpu",
    overwrite: bool = False,
    workers: int = 1,
    success_only: bool = False,
    max_attempts_per_tier: int | None = None,
) -> CollectionSummary:
    """Collect teacher labels, optionally on policy-induced DAgger states.

    ``teacher_probability=1`` records pure expert trajectories.  Lower values
    execute the neural behavior policy for the remaining decisions while still
    storing the observation-only teacher action as the supervised label. Pure
    teacher and CPU behavior-policy episodes can be distributed across
    ``workers``. Every mixed-policy episode owns a seed-derived RNG, so action
    mixtures are invariant to worker count and completion order. ``success_only``
    retains the first requested successful seeds per tier and stops with an
    auditable incomplete manifest if the deterministic attempt bound is hit.
    """
    environment_fingerprint_start = training_environment_fingerprint()
    if not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must lie in [0, 1]")
    if episodes_per_tier <= 0:
        raise ValueError("episodes_per_tier must be positive")
    workers = int(workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    tier_values = tuple(int(tier) for tier in tiers)
    if not tier_values or any(tier not in range(1, 7) for tier in tier_values):
        raise ValueError("tiers must be a non-empty subset of 1..6")
    if len(set(tier_values)) != len(tier_values):
        raise ValueError("tiers must not contain duplicates")
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    collection_device = str(torch.device(collection_device))
    uses_behavior = teacher_probability < 1.0
    if uses_behavior and behavior_checkpoint is None:
        raise ValueError("teacher_probability < 1 requires behavior_checkpoint")
    if uses_behavior and not Path(behavior_checkpoint).is_file():
        raise FileNotFoundError(f"Behavior checkpoint does not exist: {behavior_checkpoint}")
    pure_teacher = not uses_behavior
    if workers > 1 and uses_behavior and torch.device(collection_device).type != "cpu":
        raise ValueError("parallel behavior collection requires a CPU collection_device")
    if success_only and not pure_teacher:
        raise ValueError("success_only is supported only for pure-teacher collection")
    default_attempts = episodes_per_tier if not success_only else episodes_per_tier * 20
    attempt_limit = default_attempts if max_attempts_per_tier is None else int(max_attempts_per_tier)
    if attempt_limit < episodes_per_tier:
        raise ValueError("max_attempts_per_tier must be at least episodes_per_tier")
    if attempt_limit > 100_000:
        raise ValueError("max_attempts_per_tier must not exceed the per-tier seed namespace")
    max_scheduled_seed = int(seed_start + max(tier_values) * 100_000 + attempt_limit - 1)
    if max_scheduled_seed >= TRAINING_SEED_LIMIT:
        raise ValueError(
            f"imitation collection reaches seed {max_scheduled_seed}, outside the training "
            f"namespace 0..{TRAINING_SEED_LIMIT - 1}"
        )
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    behavior = (
        load_policy(Path(behavior_checkpoint), device=collection_device)
        if uses_behavior and workers == 1
        else None
    )
    total_transitions = 0
    attempted_transitions = 0
    successes = 0
    attempted_episodes = 0
    discarded_episodes = 0
    episode_records: list[dict[str, int | float | str]] = []
    tier_counts: list[dict[str, object]] = []
    shortfalls: list[dict[str, int]] = []
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        if pure_teacher:
            executor = ProcessPoolExecutor(max_workers=workers)
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_behavior_worker,
                initargs=(str(Path(behavior_checkpoint).resolve()), collection_device),
            )
    try:
        for requested_tier in tier_values:
            retained = 0
            tier_attempts = 0
            tier_discarded = 0
            tier_successes = 0
            actual_tiers: set[int] = set()
            while retained < episodes_per_tier and tier_attempts < attempt_limit:
                remaining = episodes_per_tier - retained
                batch_size = min(workers if executor is not None else 1, remaining, attempt_limit - tier_attempts)
                teacher_jobs = [
                    _PureTeacherJob(
                        output=output,
                        requested_tier=requested_tier,
                        seed=int(seed_start + requested_tier * 100_000 + tier_attempts + offset),
                        lesson=lesson,
                        success_only=success_only,
                    )
                    for offset in range(batch_size)
                ]
                if pure_teacher:
                    results = (
                        list(executor.map(_collect_pure_teacher_episode, teacher_jobs))
                        if executor is not None
                        else [_collect_pure_teacher_episode(teacher_jobs[0])]
                    )
                elif executor is not None:
                    behavior_jobs = [
                        _BehaviorJob(
                            output=job.output,
                            requested_tier=job.requested_tier,
                            seed=job.seed,
                            lesson=job.lesson,
                            teacher_probability=teacher_probability,
                        )
                        for job in teacher_jobs
                    ]
                    results = list(executor.map(_collect_behavior_episode, behavior_jobs))
                else:
                    job = teacher_jobs[0]
                    results = [
                        _collect_episode(
                            output,
                            requested_tier=job.requested_tier,
                            seed=job.seed,
                            lesson=job.lesson,
                            behavior=behavior,
                            teacher_probability=teacher_probability,
                            rng=_episode_mixture_rng(
                                requested_tier=job.requested_tier,
                                seed=job.seed,
                                lesson=job.lesson,
                            ),
                            device=collection_device,
                        )
                    ]
                tier_attempts += len(results)
                attempted_episodes += len(results)
                for result in results:
                    actual_tiers.add(result.tier)
                    attempted_transitions += result.transitions
                    if success_only and not result.behavior_success:
                        tier_discarded += 1
                        discarded_episodes += 1
                        continue
                    retained += 1
                    tier_successes += result.behavior_success
                    total_transitions += result.transitions
                    successes += result.behavior_success
                    episode_records.append(_record(result))
            tier_counts.append(
                {
                    "requested_tier": requested_tier,
                    "actual_tiers": sorted(actual_tiers),
                    "requested_episodes": episodes_per_tier,
                    "retained_episodes": retained,
                    "attempted_episodes": tier_attempts,
                    "discarded_episodes": tier_discarded,
                    "behavior_successes": tier_successes,
                }
            )
            if retained < episodes_per_tier:
                shortfalls.append(
                    {
                        "requested_tier": requested_tier,
                        "requested_episodes": episodes_per_tier,
                        "retained_episodes": retained,
                    }
                )
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    environment_fingerprint_end = training_environment_fingerprint()
    environment_stable = environment_fingerprint_start == environment_fingerprint_end
    manifest = {
        "format": "ghostline-imitation-v2",
        "observation_contract": "GhostlineEnv-v2",
        "teacher": "ObservationTeacherPolicy",
        "teacher_uses_privileged_state": False,
        "seed_start": seed_start,
        "seed_contract": "seed_start + requested_tier * 100000 + attempt_index",
        "max_scheduled_seed": max_scheduled_seed,
        "seed_namespace": f"training:0..{TRAINING_SEED_LIMIT - 1}",
        "lesson": lesson,
        "teacher_probability": teacher_probability,
        "behavior_checkpoint": str(behavior_checkpoint) if behavior_checkpoint else None,
        "behavior_policy_used": uses_behavior,
        "collection_device": collection_device,
        "workers": workers,
        "success_only": success_only,
        "max_attempts_per_tier": attempt_limit,
        "complete": not shortfalls and environment_stable,
        "environment_fingerprint": environment_fingerprint_start,
        "environment_stable_during_collection": environment_stable,
        "episodes": len(episode_records),
        "attempted_episodes": attempted_episodes,
        "discarded_episodes": discarded_episodes,
        "transitions": total_transitions,
        "attempted_transitions": attempted_transitions,
        "discarded_transitions": attempted_transitions - total_transitions,
        "behavior_successes": successes,
        "behavior_success_rate": successes / max(1, len(episode_records)),
        "attempt_success_rate": successes / max(1, attempted_episodes),
        "tier_counts": tier_counts,
        "shortfalls": shortfalls,
        "records": episode_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = CollectionSummary(
        len(episode_records),
        total_transitions,
        successes,
        output,
        attempted_episodes,
        discarded_episodes,
    )
    if not environment_stable:
        raise RuntimeError(
            "Ghostline environment files changed during trajectory collection; "
            f"the dataset is invalid and preserved only for audit at {output}"
        )
    if shortfalls:
        shortfall = shortfalls[0]
        raise RuntimeError(
            "success-only collection exhausted its deterministic attempt limit for "
            f"requested tier {shortfall['requested_tier']}: retained "
            f"{shortfall['retained_episodes']}/{shortfall['requested_episodes']}; "
            f"see {output / 'manifest.json'}"
        )
    return summary


def _episode_tier(path: Path) -> int:
    match = _EPISODE_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"Invalid imitation episode filename: {path}")
    tier = int(match.group(1))
    if tier not in range(1, 7):
        raise RuntimeError(f"Invalid tier in imitation episode filename: {path}")
    return tier


def _build_episode_split(
    roots: tuple[Path, ...],
    files_by_root: tuple[tuple[Path, ...], ...],
    *,
    validation_fraction: float,
    split_seed: int,
) -> tuple[tuple[tuple[Path, ...], ...], tuple[tuple[Path, ...], ...], dict[str, object]]:
    train_by_root: list[tuple[Path, ...]] = []
    validation_by_root: list[tuple[Path, ...]] = []
    assignments: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    for root_index, (root, root_files) in enumerate(zip(roots, files_by_root)):
        train_files: list[Path] = []
        validation_files: list[Path] = []
        by_tier: dict[int, list[Path]] = {}
        for path in root_files:
            by_tier.setdefault(_episode_tier(path), []).append(path)
        for tier, tier_files in sorted(by_tier.items()):
            ranked = sorted(
                tier_files,
                key=lambda path: hashlib.sha256(
                    f"{split_seed}:{root_index}:{tier}:{path.name}".encode("utf-8")
                ).digest(),
            )
            if len(ranked) >= 2:
                validation_count = min(
                    len(ranked) - 1,
                    max(1, int(round(len(ranked) * validation_fraction))),
                )
            else:
                validation_count = 0
            validation_names = {path.name for path in ranked[:validation_count]}
            for path in sorted(tier_files):
                assignment = "validation" if path.name in validation_names else "train"
                (validation_files if assignment == "validation" else train_files).append(path)
                assignments.append(
                    {
                        "root_index": root_index,
                        "tier": tier,
                        "episode": path.name,
                        "split": assignment,
                    }
                )
            groups.append(
                {
                    "root_index": root_index,
                    "root": str(root),
                    "tier": tier,
                    "episodes": len(tier_files),
                    "train_episodes": len(tier_files) - validation_count,
                    "validation_episodes": validation_count,
                    "representative": len(tier_files) >= 2,
                }
            )
        train_by_root.append(tuple(sorted(train_files)))
        validation_by_root.append(tuple(sorted(validation_files)))
    canonical_assignments = sorted(
        assignments,
        key=lambda item: (int(item["root_index"]), int(item["tier"]), str(item["episode"])),
    )
    split_digest = hashlib.sha256(
        json.dumps(
            {
                "format": "ghostline-episode-split-v1",
                "split_seed": int(split_seed),
                "validation_fraction": float(validation_fraction),
                "assignments": canonical_assignments,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report: dict[str, object] = {
        "format": "ghostline-episode-split-v1",
        "split_seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "split_digest": split_digest,
        "total_episodes": len(canonical_assignments),
        "train_episodes": sum(len(files) for files in train_by_root),
        "validation_episodes": sum(len(files) for files in validation_by_root),
        "unsplittable_groups": sum(not bool(group["representative"]) for group in groups),
        "groups": groups,
        "assignments": canonical_assignments,
    }
    return tuple(train_by_root), tuple(validation_by_root), report


class EpisodeSequenceDataset:
    """Strict episode split plus deterministic stratified recurrent windows."""

    def __init__(
        self,
        roots: Iterable[Path],
        *,
        cache_size: int = 12,
        latest_root_fraction: float | None = None,
        split: str = "all",
        validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
        split_seed: int = 0,
        require_representative_split: bool = False,
    ) -> None:
        self.roots = tuple(Path(root) for root in roots)
        if split not in {"all", "train", "validation"}:
            raise ValueError("split must be one of: all, train, validation")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        expected_fingerprint = training_environment_fingerprint()
        for root in self.roots:
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Ghostline imitation dataset {root} has no manifest.json; "
                    "unversioned trajectories are not eligible for training"
                )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError(f"Invalid imitation manifest at {manifest_path}") from exc
            if manifest.get("format") != "ghostline-imitation-v2":
                raise RuntimeError(f"Unsupported imitation dataset format at {manifest_path}")
            if manifest.get("observation_contract") != OBSERVATION_CONTRACT:
                raise RuntimeError(f"Stale observation contract in {manifest_path}")
            if not manifest.get("complete") or not manifest.get("environment_stable_during_collection"):
                raise RuntimeError(f"Incomplete or unstable imitation dataset at {manifest_path}")
            actual_fingerprint = manifest.get("environment_fingerprint")
            if actual_fingerprint != expected_fingerprint:
                raise RuntimeError(
                    f"Stale imitation dataset at {manifest_path}: "
                    f"{str(actual_fingerprint)[:12]} != {expected_fingerprint[:12]}"
                )
        all_files_by_root = tuple(
            tuple(sorted(root.glob("tier-*-seed-*.npz"))) for root in self.roots
        )
        episode_locations: dict[str, Path] = {}
        for root_files in all_files_by_root:
            for path in root_files:
                identity = path.name
                previous = episode_locations.get(identity)
                if previous is not None:
                    raise RuntimeError(
                        "Duplicate imitation episode identity across roots: "
                        f"{identity} appears in {previous.parent} and {path.parent}"
                    )
                episode_locations[identity] = path
        train_by_root, validation_by_root, split_report = _build_episode_split(
            self.roots,
            all_files_by_root,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        if require_representative_split and int(split_report["unsplittable_groups"]):
            unsplittable = [
                f"root={group['root_index']},tier={group['tier']}"
                for group in split_report["groups"]
                if not group["representative"]
            ]
            raise RuntimeError(
                "Leakage-free root/tier validation requires at least two episodes per "
                f"group; unsplittable groups: {', '.join(unsplittable)}"
            )
        if split == "train":
            selected_by_root = train_by_root
        elif split == "validation":
            selected_by_root = validation_by_root
        else:
            selected_by_root = all_files_by_root
        self.all_files_by_root = all_files_by_root
        self.files_by_root = selected_by_root
        self.files = [path for files in self.files_by_root for path in files]
        if not self.files:
            if split == "validation":
                raise FileNotFoundError(
                    "No held-out imitation episodes were available; every root/tier group "
                    "needs at least two episodes for leakage-free validation"
                )
            raise FileNotFoundError(f"No Ghostline imitation episode files were found for split={split}")
        if latest_root_fraction is not None and not 0.0 <= latest_root_fraction <= 1.0:
            raise ValueError("latest_root_fraction must lie in [0, 1]")
        if latest_root_fraction is not None and len(self.files_by_root) > 1:
            if not self.files_by_root[-1]:
                raise FileNotFoundError(f"The latest imitation dataset root has no {split} episodes")
            if not any(self.files_by_root[:-1]):
                raise FileNotFoundError(f"Prior imitation dataset roots have no {split} episodes")
        self.split = split
        self.split_report = {
            **split_report,
            "selected_split": split,
            "selected_episodes": len(self.files),
        }
        self.file_ids = tuple(
            f"{root_index}:{path.name}"
            for root_index, files in enumerate(self.files_by_root)
            for path in files
        )
        self._root_index_by_path = {
            path: root_index
            for root_index, files in enumerate(self.files_by_root)
            for path in files
        }
        self.latest_root_fraction = latest_root_fraction
        self.prior_files = [path for files in self.files_by_root[:-1] for path in files]
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._window_candidates: dict[Path, dict[str, np.ndarray]] | None = None
        self._eligible_by_root: dict[str, tuple[tuple[Path, ...], ...]] = {}
        self._corpus_window_metrics: dict[str, object] | None = None
        self.last_sample_metrics: dict[str, object] = {}

    def _sample_path(self, rng: np.random.Generator) -> Path:
        if self.latest_root_fraction is not None and len(self.files_by_root) > 1:
            source = self.files_by_root[-1] if rng.random() < self.latest_root_fraction else self.prior_files
        else:
            source = self.files
        return source[int(rng.integers(0, len(source)))]

    def _sample_eligible_path(
        self, stratum: str, rng: np.random.Generator
    ) -> Path | None:
        eligible_by_root = self._eligible_by_root[stratum]
        if self.latest_root_fraction is not None and len(eligible_by_root) > 1:
            latest = eligible_by_root[-1]
            prior = tuple(path for files in eligible_by_root[:-1] for path in files)
            prefer_latest = rng.random() < self.latest_root_fraction
            source = latest if prefer_latest else prior
            if not source:
                source = prior if prefer_latest else latest
        else:
            source = tuple(path for files in eligible_by_root for path in files)
        if not source:
            return None
        return source[int(rng.integers(0, len(source)))]

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.pop(path, None)
        if cached is None:
            with np.load(path, allow_pickle=False) as data:
                cached = {key: data[key] for key in data.files}
        self._cache[path] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def _ensure_window_index(self) -> None:
        if self._window_candidates is not None:
            return
        candidates_by_path: dict[Path, dict[str, np.ndarray]] = {}
        eligible: dict[str, list[list[Path]]] = {
            stratum: [[] for _ in self.roots] for stratum in WINDOW_STRATA[1:]
        }
        candidate_states = {stratum: 0 for stratum in WINDOW_STRATA[1:]}
        transitions = 0
        for path in self.files:
            with np.load(path, allow_pickle=False) as data:
                actions = np.asarray(data["action"], dtype=np.int64)
                behavior_actions = np.asarray(
                    data["behavior_action"] if "behavior_action" in data else actions,
                    dtype=np.int64,
                )
            if actions.ndim != 1 or len(actions) == 0 or behavior_actions.shape != actions.shape:
                raise RuntimeError(f"Invalid action arrays in imitation episode: {path}")
            path_candidates = {
                "endpoint": np.asarray([len(actions) - 1], dtype=np.int64),
                "action_change": np.flatnonzero(actions[1:] != actions[:-1]).astype(np.int64) + 1,
                "dash": np.flatnonzero((actions // 9) % 2 == 1).astype(np.int64),
                "pulse": np.flatnonzero(actions // 18 == 1).astype(np.int64),
                "recovery": np.flatnonzero(actions != behavior_actions).astype(np.int64),
            }
            candidates_by_path[path] = path_candidates
            transitions += len(actions)
            root_index = self._root_index_by_path[path]
            for stratum, indices in path_candidates.items():
                candidate_states[stratum] += len(indices)
                if len(indices):
                    eligible[stratum][root_index].append(path)
        self._window_candidates = candidates_by_path
        self._eligible_by_root = {
            stratum: tuple(tuple(files) for files in files_by_root)
            for stratum, files_by_root in eligible.items()
        }
        self._corpus_window_metrics = {
            "episodes": len(self.files),
            "transitions": transitions,
            "candidate_states": candidate_states,
            "candidate_episodes": {
                stratum: sum(len(files) for files in self._eligible_by_root[stratum])
                for stratum in WINDOW_STRATA[1:]
            },
        }

    def corpus_window_metrics(self) -> dict[str, object]:
        self._ensure_window_index()
        assert self._corpus_window_metrics is not None
        return json.loads(json.dumps(self._corpus_window_metrics))

    @staticmethod
    def _window_schedule(
        batch_size: int,
        *,
        stratified: bool,
        uniform_fraction: float,
        rng: np.random.Generator,
    ) -> list[str]:
        if not stratified:
            return ["uniform"] * batch_size
        uniform_count = int(round(batch_size * uniform_fraction))
        if uniform_fraction > 0.0 and uniform_count == 0:
            uniform_count = 1
        uniform_count = min(batch_size, uniform_count)
        schedule = ["uniform"] * uniform_count
        remaining = batch_size - uniform_count
        specialized = list(WINDOW_STRATA[1:])
        raw_counts = np.asarray(
            [remaining * SPECIALIZED_WINDOW_WEIGHTS[stratum] for stratum in specialized]
        )
        counts = np.floor(raw_counts).astype(np.int64)
        leftover = remaining - int(counts.sum())
        if leftover:
            order = sorted(
                range(len(specialized)),
                key=lambda index: (
                    -(raw_counts[index] - counts[index]),
                    SPECIALIZED_TIE_ORDER.index(specialized[index]),
                ),
            )
            for index in order[:leftover]:
                counts[index] += 1
        for stratum, count in zip(specialized, counts):
            schedule.extend([stratum] * int(count))
        rng.shuffle(schedule)
        return schedule

    def _training_start(
        self,
        path: Path,
        *,
        length: int,
        sequence_length: int,
        stratum: str,
        rng: np.random.Generator,
    ) -> tuple[int, int | None]:
        max_start = max(0, length - sequence_length)
        if stratum == "uniform":
            return int(rng.integers(0, max_start + 1)), None
        assert self._window_candidates is not None
        candidates = self._window_candidates[path][stratum]
        anchor = int(candidates[int(rng.integers(0, len(candidates)))])
        if stratum == "endpoint":
            return max_start, anchor
        low = max(0, anchor - sequence_length + 1)
        high = min(anchor, max_start)
        return int(rng.integers(low, high + 1)), anchor

    def sample(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        burn_in: int = 0,
        rng: np.random.Generator,
        device: torch.device,
        stratified: bool = True,
        uniform_fraction: float = DEFAULT_UNIFORM_WINDOW_FRACTION,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must lie in [0, 1]")
        self._ensure_window_index()
        schedule = self._window_schedule(
            batch_size,
            stratified=stratified,
            uniform_fraction=uniform_fraction,
            rng=rng,
        )
        obs_batches: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
        targets: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "action",
                "behavior_action",
                "return_",
                "objective_bearing",
                "danger",
                "valid",
                "anchor_weight",
            )
        }
        requested_counts = {stratum: 0 for stratum in WINDOW_STRATA}
        actual_counts = {stratum: 0 for stratum in WINDOW_STRATA}
        fallbacks: dict[str, int] = {}
        episode_ids: list[str] = []
        training_starts: list[int] = []
        anchors: list[int | None] = []
        actual_strata: list[str] = []
        total_length = sequence_length + burn_in
        for requested_stratum in schedule:
            requested_counts[requested_stratum] += 1
            actual_stratum = requested_stratum
            if requested_stratum == "uniform":
                path = self._sample_path(rng)
            else:
                path = self._sample_eligible_path(requested_stratum, rng)
                if path is None:
                    path = self._sample_path(rng)
                    actual_stratum = "uniform"
                    fallback = f"{requested_stratum}->uniform"
                    fallbacks[fallback] = fallbacks.get(fallback, 0) + 1
            actual_counts[actual_stratum] += 1
            episode = self._load(path)
            length = len(episode["action"])
            training_start, anchor = self._training_start(
                path,
                length=length,
                sequence_length=sequence_length,
                stratum=actual_stratum,
                rng=rng,
            )
            start = max(0, training_start - burn_in)
            loss_offset = training_start - start
            count = min(sequence_length, length - training_start)
            indices = np.minimum(np.arange(start, start + total_length), length - 1)
            for key in OBS_KEYS:
                obs_batches[key].append(episode[key][indices])
            for key in ("action", "return_", "objective_bearing", "danger"):
                targets[key].append(episode[key][indices])
            behavior_actions = episode.get("behavior_action", episode["action"])
            targets["behavior_action"].append(behavior_actions[indices])
            valid = np.zeros(total_length, dtype=np.float32)
            valid[loss_offset : loss_offset + count] = 1.0
            targets["valid"].append(valid)
            anchor_weight = np.ones(total_length, dtype=np.float32)
            if anchor is not None:
                anchor_position = anchor - start
                if not valid[anchor_position]:
                    raise RuntimeError("Stratified anchor fell outside supervised timesteps")
                anchor_weight[anchor_position] = ANCHOR_LOSS_WEIGHTS[actual_stratum]
            targets["anchor_weight"].append(anchor_weight)
            root_index = self._root_index_by_path[path]
            episode_ids.append(f"{root_index}:{path.name}")
            training_starts.append(training_start)
            anchors.append(anchor)
            actual_strata.append(actual_stratum)
        observations = {
            key: torch.as_tensor(np.stack(values, axis=1), device=device)
            for key, values in obs_batches.items()
        }
        labels = {
            key: torch.as_tensor(np.stack(values, axis=1), device=device)
            for key, values in targets.items()
        }
        self.last_sample_metrics = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "burn_in": burn_in,
            "stratified": stratified,
            "uniform_fraction": uniform_fraction,
            "requested_counts": requested_counts,
            "actual_counts": actual_counts,
            "fallbacks": fallbacks,
            "episode_ids": episode_ids,
            "training_starts": training_starts,
            "anchors": anchors,
            "actual_strata": actual_strata,
        }
        return observations, labels


def recovery_supervision_weights(
    teacher_actions: torch.Tensor,
    behavior_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 4x weights for policy-induced states requiring a move correction."""
    recovery = teacher_actions.long().remainder(9) != behavior_actions.long().remainder(9)
    return 1.0 + 3.0 * recovery.float(), recovery.float()


def behavior_clone_losses(
    logits: torch.Tensor,
    values: torch.Tensor,
    bearing: torch.Tensor,
    danger: torch.Tensor,
    labels: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute the weighted joint, factorized, and player-equivalent BC losses."""
    valid = labels["valid"].float()
    denominator = valid.sum().clamp_min(1.0)
    anchor_weight = labels.get("anchor_weight")
    if anchor_weight is None:
        anchor_weight = torch.ones_like(valid)
    else:
        anchor_weight = anchor_weight.float().clamp(1.0, max(ANCHOR_LOSS_WEIGHTS.values()))
    supervision_weight, move_recovery = recovery_supervision_weights(
        labels["action"], labels["behavior_action"]
    )
    priority_weight = torch.maximum(supervision_weight, anchor_weight)
    weighted_valid = valid * priority_weight
    weighted_denominator = weighted_valid.sum().clamp_min(1.0)
    imitation = nn.functional.cross_entropy(
        logits.flatten(0, 1), labels["action"].long().flatten(), reduction="none"
    ).reshape_as(valid)
    imitation_loss = (imitation * weighted_valid).sum() / weighted_denominator
    move_nll, dash_nll, pulse_nll = factorized_action_nll(logits, labels["action"])
    move_loss = (move_nll * weighted_valid).sum() / weighted_denominator
    dash_loss = (dash_nll * weighted_valid).sum() / weighted_denominator
    pulse_loss = (pulse_nll * weighted_valid).sum() / weighted_denominator
    factorized_loss = (move_loss + dash_loss + pulse_loss) / 3.0
    bearing_loss = (
        nn.functional.smooth_l1_loss(
            bearing, labels["objective_bearing"], reduction="none"
        ).mean(-1)
        * valid
    ).sum() / denominator
    danger_loss = (
        nn.functional.binary_cross_entropy(danger, labels["danger"], reduction="none")
        * valid
    ).sum() / denominator
    value_error = nn.functional.smooth_l1_loss(values, labels["return_"], reduction="none")
    value_loss = (value_error * valid).sum() / denominator
    loss = (
        imitation_loss
        + 0.25 * factorized_loss
        + 0.20 * bearing_loss
        + 0.10 * danger_loss
        + 0.03 * value_loss
    )
    predicted_actions = logits.argmax(-1)
    accuracy = ((predicted_actions == labels["action"]) * valid).sum() / denominator
    predicted_move = predicted_actions.remainder(9)
    teacher_move = labels["action"].long().remainder(9)
    predicted_dash = predicted_actions.div(9, rounding_mode="floor").remainder(2)
    teacher_dash = labels["action"].long().div(9, rounding_mode="floor").remainder(2)
    predicted_pulse = predicted_actions.div(18, rounding_mode="floor")
    teacher_pulse = labels["action"].long().div(18, rounding_mode="floor")
    move_accuracy = ((predicted_move == teacher_move) * valid).sum() / denominator
    dash_accuracy = ((predicted_dash == teacher_dash) * valid).sum() / denominator
    pulse_accuracy = ((predicted_pulse == teacher_pulse) * valid).sum() / denominator
    dash_positive_valid = (teacher_dash == 1).float() * valid
    pulse_positive_valid = (teacher_pulse == 1).float() * valid
    dash_positive_count = dash_positive_valid.sum()
    pulse_positive_count = pulse_positive_valid.sum()
    dash_predicted_positive = (predicted_dash == 1).float() * valid
    pulse_predicted_positive = (predicted_pulse == 1).float() * valid
    dash_positive_recall = (
        ((predicted_dash == 1) * dash_positive_valid).sum()
        / dash_positive_count.clamp_min(1.0)
    )
    pulse_positive_recall = (
        ((predicted_pulse == 1) * pulse_positive_valid).sum()
        / pulse_positive_count.clamp_min(1.0)
    )
    dash_positive_precision = (
        ((teacher_dash == 1) * dash_predicted_positive).sum()
        / dash_predicted_positive.sum().clamp_min(1.0)
    )
    pulse_positive_precision = (
        ((teacher_pulse == 1) * pulse_predicted_positive).sum()
        / pulse_predicted_positive.sum().clamp_min(1.0)
    )
    recovery = (labels["action"].long() != labels["behavior_action"].long()).float()
    recovery_valid = recovery * valid
    recovery_count = recovery_valid.sum()
    recovery_action_accuracy = (
        ((predicted_actions == labels["action"]) * recovery_valid).sum()
        / recovery_count.clamp_min(1.0)
    )
    recovery_move_accuracy = (
        ((predicted_move == teacher_move) * recovery_valid).sum()
        / recovery_count.clamp_min(1.0)
    )
    move_recovery_valid = move_recovery * valid
    move_recovery_count = move_recovery_valid.sum()
    move_correction_accuracy = (
        ((predicted_move == teacher_move) * move_recovery_valid).sum()
        / move_recovery_count.clamp_min(1.0)
    )
    return {
        "loss": loss,
        "imitation_loss": imitation_loss,
        "factorized_action_loss": factorized_loss,
        "move_nll": move_loss,
        "dash_nll": dash_loss,
        "pulse_nll": pulse_loss,
        "bearing_loss": bearing_loss,
        "danger_loss": danger_loss,
        "value_loss": value_loss,
        "action_accuracy": accuracy,
        "move_accuracy": move_accuracy,
        "dash_accuracy": dash_accuracy,
        "pulse_accuracy": pulse_accuracy,
        "dash_positive_count": dash_positive_count,
        "dash_positive_precision": dash_positive_precision,
        "dash_positive_recall": dash_positive_recall,
        "pulse_positive_count": pulse_positive_count,
        "pulse_positive_precision": pulse_positive_precision,
        "pulse_positive_recall": pulse_positive_recall,
        "mean_priority_weight": weighted_valid.sum() / denominator,
        "recovery_fraction": recovery_count / denominator,
        "recovery_action_accuracy": recovery_action_accuracy,
        "recovery_move_accuracy": recovery_move_accuracy,
        "move_recovery_fraction": move_recovery_count / denominator,
        "move_correction_accuracy": move_correction_accuracy,
    }


def _json_metric_values(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    names = (
        "loss",
        "imitation_loss",
        "factorized_action_loss",
        "move_nll",
        "dash_nll",
        "pulse_nll",
        "action_accuracy",
        "move_accuracy",
        "dash_accuracy",
        "dash_positive_count",
        "dash_positive_precision",
        "dash_positive_recall",
        "pulse_accuracy",
        "pulse_positive_count",
        "pulse_positive_precision",
        "pulse_positive_recall",
        "mean_priority_weight",
        "recovery_fraction",
        "recovery_action_accuracy",
        "recovery_move_accuracy",
        "move_recovery_fraction",
        "move_correction_accuracy",
    )
    return {name: float(metrics[name].detach().cpu()) for name in names}


def train_behavior_clone(
    dataset_roots: Iterable[Path],
    output: Path,
    *,
    updates: int,
    batch_size: int = 16,
    sequence_length: int = DEFAULT_BC_SEQUENCE_LENGTH,
    burn_in: int = DEFAULT_BC_BURN_IN,
    learning_rate: float = 2e-4,
    recurrent: bool = True,
    recurrent_size: int = 384,
    seed: int = 0,
    device: str | torch.device = "cpu",
    resume: bool = True,
    init_checkpoint: Path | None = None,
    latest_root_fraction: float | None = None,
    validation_interval: int = 100,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = 0,
    uniform_window_fraction: float = DEFAULT_UNIFORM_WINDOW_FRACTION,
    validation_windows: int = DEFAULT_VALIDATION_WINDOWS,
) -> Path:
    if updates <= 0:
        raise ValueError("updates must be positive")
    if validation_interval <= 0:
        raise ValueError("validation_interval must be positive")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if not 0.0 <= uniform_window_fraction <= 1.0:
        raise ValueError("uniform_window_fraction must lie in [0, 1]")
    if validation_windows < DEFAULT_VALIDATION_WINDOWS:
        raise ValueError(
            f"validation_windows must be at least {DEFAULT_VALIDATION_WINDOWS}"
        )
    device_value = torch.device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    roots = tuple(dataset_roots)
    dataset = EpisodeSequenceDataset(
        roots,
        latest_root_fraction=latest_root_fraction,
        split="train",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        require_representative_split=True,
    )
    validation_dataset = EpisodeSequenceDataset(
        roots,
        latest_root_fraction=latest_root_fraction,
        split="validation",
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        require_representative_split=True,
    )
    overlap = set(dataset.file_ids) & set(validation_dataset.file_ids)
    if overlap:
        raise RuntimeError(f"Imitation train/validation episode leakage detected: {sorted(overlap)[:3]}")
    validation_observations, validation_labels = validation_dataset.sample(
        batch_size=max(validation_windows, batch_size),
        sequence_length=sequence_length,
        burn_in=burn_in,
        rng=np.random.default_rng(seed + 1_000_003),
        device=torch.device("cpu"),
        uniform_fraction=uniform_window_fraction,
    )
    validation_input_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*validation_observations.values(), *validation_labels.values())
    )
    output.mkdir(parents=True, exist_ok=True)
    split_report = {
        "format": "ghostline-bc-data-v1",
        "window_sampler_version": WINDOW_SAMPLER_VERSION,
        "specialized_window_weights": SPECIALIZED_WINDOW_WEIGHTS,
        "anchor_loss_weights": ANCHOR_LOSS_WEIGHTS,
        "split": dataset.split_report,
        "train_file_ids": list(dataset.file_ids),
        "validation_file_ids": list(validation_dataset.file_ids),
        "train_window_candidates": dataset.corpus_window_metrics(),
        "validation_window_candidates": validation_dataset.corpus_window_metrics(),
        "held_out_validation_sample": validation_dataset.last_sample_metrics,
        "sequence_length": sequence_length,
        "burn_in": burn_in,
        "uniform_window_fraction": uniform_window_fraction,
        "validation_windows": max(validation_windows, batch_size),
        "validation_input_tensor_bytes": validation_input_bytes,
        "validation_input_tensor_mib": validation_input_bytes / (1024.0**2),
        "latest_root_fraction": latest_root_fraction,
    }
    (output / "data-split.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    split_digest = str(dataset.split_report["split_digest"])
    checkpoint_path = output / "latest.pt"
    if init_checkpoint is not None:
        policy = load_policy(init_checkpoint, device=device_value)
        if policy.recurrent != recurrent or policy.recurrent_size != recurrent_size:
            raise ValueError("initial imitation checkpoint architecture does not match requested recurrence")
        policy.train()
    else:
        policy = UniversalGhostlinePolicy(recurrent=recurrent, recurrent_size=recurrent_size).to(device_value)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, eps=1e-5, weight_decay=1e-4)
    start_update = 0
    best_validation_loss = float("inf")
    last_validation_loss = float("inf")
    if resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device_value, weights_only=False)
        require_current_checkpoint(payload, path=checkpoint_path)
        expected_resume_contract = {
            "data_split_digest": split_digest,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "burn_in": burn_in,
            "uniform_window_fraction": uniform_window_fraction,
            "validation_windows": max(validation_windows, batch_size),
            "window_sampler_version": WINDOW_SAMPLER_VERSION,
            "specialized_window_weights": SPECIALIZED_WINDOW_WEIGHTS,
            "anchor_loss_weights": ANCHOR_LOSS_WEIGHTS,
        }
        for name, expected in expected_resume_contract.items():
            if payload.get(name) != expected:
                raise RuntimeError(
                    f"Cannot resume behavior cloning with changed {name}: "
                    f"checkpoint={payload.get(name)!r}, requested={expected!r}"
                )
        policy.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        rng_state = payload.get("numpy_rng_state")
        if rng_state is None:
            raise RuntimeError("Cannot resume behavior cloning without numpy_rng_state")
        rng.bit_generator.state = rng_state
        torch_rng_state = payload.get("torch_rng_state")
        if torch_rng_state is None:
            raise RuntimeError("Cannot resume behavior cloning without torch_rng_state")
        torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_states = payload.get("cuda_rng_states")
        if device_value.type == "cuda" and cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        start_update = int(payload.get("update", 0))
        best_validation_loss = float(
            payload.get("best_validation_loss", payload.get("best_loss", best_validation_loss))
        )
        last_validation_loss = float(payload.get("validation_loss", last_validation_loss))
    writer = SummaryWriter(output / "tensorboard")
    for update in range(start_update, updates):
        observations, labels = dataset.sample(
            batch_size=batch_size,
            sequence_length=sequence_length,
            burn_in=burn_in,
            rng=rng,
            device=device_value,
            uniform_fraction=uniform_window_fraction,
        )
        logits, values, bearing, danger, _ = policy.forward_sequence_aux(observations)
        metrics = behavior_clone_losses(logits, values, bearing, danger, labels)
        loss = metrics["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        for name, value in metrics.items():
            writer.add_scalar(f"bc/{name}", float(value.detach()), update + 1)
        window_metrics = dataset.last_sample_metrics
        actual_counts = window_metrics["actual_counts"]
        assert isinstance(actual_counts, dict)
        for stratum in WINDOW_STRATA:
            writer.add_scalar(
                f"bc_windows/{stratum}_fraction",
                float(actual_counts[stratum]) / batch_size,
                update + 1,
            )
        fallbacks = window_metrics["fallbacks"]
        assert isinstance(fallbacks, dict)
        writer.add_scalar(
            "bc_windows/fallback_fraction",
            sum(int(value) for value in fallbacks.values()) / batch_size,
            update + 1,
        )

        validate = (update + 1) % validation_interval == 0 or update + 1 == updates
        if validate:
            policy.eval()
            with torch.no_grad():
                validation_device_observations = {
                    key: value.to(device_value) for key, value in validation_observations.items()
                }
                validation_device_labels = {
                    key: value.to(device_value) for key, value in validation_labels.items()
                }
                validation_outputs = policy.forward_sequence_aux(
                    validation_device_observations
                )
                validation_metrics = behavior_clone_losses(
                    *validation_outputs[:4], validation_device_labels
                )
                del validation_device_observations, validation_device_labels, validation_outputs
            policy.train()
            last_validation_loss = float(validation_metrics["loss"])
            for name, value in validation_metrics.items():
                writer.add_scalar(f"bc_validation/{name}", float(value), update + 1)
            if last_validation_loss < best_validation_loss:
                best_validation_loss = last_validation_loss
                save_policy(
                    policy,
                    output / "best.pt",
                    stage="behavior_cloning_held_out_episode_validation",
                    updates=update + 1,
                    validation_loss=best_validation_loss,
                    validation_action_accuracy=float(validation_metrics["action_accuracy"]),
                    validation_recovery_move_accuracy=float(
                        validation_metrics["recovery_move_accuracy"]
                    ),
                    data_split_digest=split_digest,
                    train_episodes=len(dataset.files),
                    validation_episodes=len(validation_dataset.files),
                    sequence_length=sequence_length,
                    burn_in=burn_in,
                    uniform_window_fraction=uniform_window_fraction,
                    validation_windows=max(validation_windows, batch_size),
                    window_sampler_version=WINDOW_SAMPLER_VERSION,
                    specialized_window_weights=SPECIALIZED_WINDOW_WEIGHTS,
                    anchor_loss_weights=ANCHOR_LOSS_WEIGHTS,
                )
            torch.save(
                {
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "recurrent": recurrent,
                    "recurrent_size": recurrent_size,
                    "update": update + 1,
                    "best_validation_loss": best_validation_loss,
                    "validation_loss": last_validation_loss,
                    "latest_root_fraction": latest_root_fraction,
                    "data_split_digest": split_digest,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "burn_in": burn_in,
                    "uniform_window_fraction": uniform_window_fraction,
                    "validation_windows": max(validation_windows, batch_size),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all() if device_value.type == "cuda" else None
                    ),
                    "window_sampler_version": WINDOW_SAMPLER_VERSION,
                    "specialized_window_weights": SPECIALIZED_WINDOW_WEIGHTS,
                    "anchor_loss_weights": ANCHOR_LOSS_WEIGHTS,
                    "observation_contract": OBSERVATION_CONTRACT,
                    "environment_fingerprint": training_environment_fingerprint(),
                },
                checkpoint_path,
            )
            target_window_fractions = {
                "uniform": uniform_window_fraction,
                **{
                    stratum: (1.0 - uniform_window_fraction) * weight
                    for stratum, weight in SPECIALIZED_WINDOW_WEIGHTS.items()
                },
            }
            training_state = {
                "format": "ghostline-bc-training-state-v1",
                "update": update + 1,
                "target_updates": updates,
                "checkpoint": str(checkpoint_path),
                "best_checkpoint": str(output / "best.pt"),
                "best_validation_loss": best_validation_loss,
                "data_split_digest": split_digest,
                "window_sampler_version": WINDOW_SAMPLER_VERSION,
                "anchor_loss_weights": ANCHOR_LOSS_WEIGHTS,
                "split_sizes": {
                    "train_episodes": len(dataset.files),
                    "held_out_episodes": len(validation_dataset.files),
                },
                "recurrent_window": {
                    "sequence_length": sequence_length,
                    "burn_in": burn_in,
                    "validation_windows": max(validation_windows, batch_size),
                    "validation_input_tensor_bytes": validation_input_bytes,
                    "validation_input_tensor_mib": validation_input_bytes / (1024.0**2),
                    "target_fractions": target_window_fractions,
                    "last_train_requested_counts": window_metrics["requested_counts"],
                    "last_train_actual_counts": window_metrics["actual_counts"],
                    "last_train_fallbacks": window_metrics["fallbacks"],
                    "held_out_requested_counts": validation_dataset.last_sample_metrics[
                        "requested_counts"
                    ],
                    "held_out_actual_counts": validation_dataset.last_sample_metrics[
                        "actual_counts"
                    ],
                    "held_out_fallbacks": validation_dataset.last_sample_metrics["fallbacks"],
                },
                "train": _json_metric_values(metrics),
                "held_out": _json_metric_values(validation_metrics),
            }
            training_state_path = output / "training-state.json"
            temporary_state_path = output / "training-state.json.tmp"
            temporary_state_path.write_text(
                json.dumps(training_state, indent=2), encoding="utf-8"
            )
            temporary_state_path.replace(training_state_path)
    writer.close()
    return checkpoint_path


def run_dagger(
    base_dataset: Path,
    output: Path,
    initial_checkpoint: Path,
    *,
    rounds: int,
    episodes_per_tier: int,
    updates_per_round: int,
    start_round: int = 1,
    beta_start: float = 0.5,
    beta_decay: float = 0.5,
    recurrent_size: int = 384,
    collection_device: str | torch.device = "cpu",
    training_device: str | torch.device | None = None,
    collection_workers: int = DEFAULT_DAGGER_COLLECTION_WORKERS,
    device: str | torch.device | None = None,
    sequence_length: int = DEFAULT_BC_SEQUENCE_LENGTH,
    burn_in: int = DEFAULT_BC_BURN_IN,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    split_seed: int = 0,
    uniform_window_fraction: float = DEFAULT_UNIFORM_WINDOW_FRACTION,
    validation_windows: int = DEFAULT_VALIDATION_WINDOWS,
) -> Path:
    """Run numbered DAgger rounds, reusing data from completed earlier rounds.

    ``rounds`` is the final round number, preserving the original behavior when
    ``start_round`` is one. A resumed run starts at ``start_round`` and adds all
    ``output/round-N/data`` directories before it to the base dataset.
    """
    if start_round < 1:
        raise ValueError("start_round must be at least 1")
    if rounds < start_round:
        raise ValueError("rounds must be greater than or equal to start_round")
    if not 0.0 <= beta_start <= 1.0:
        raise ValueError("beta_start must lie in [0, 1]")
    if not 0.0 <= beta_decay <= 1.0:
        raise ValueError("beta_decay must lie in [0, 1]")
    if episodes_per_tier <= 0 or episodes_per_tier > DAGGER_ROUND_SEED_STRIDE:
        raise ValueError(
            f"episodes_per_tier must lie in 1..{DAGGER_ROUND_SEED_STRIDE} "
            "to keep DAgger round seed subranges disjoint"
        )
    collection_workers = int(collection_workers)
    if collection_workers <= 0:
        raise ValueError("collection_workers must be positive")
    collection_device = str(torch.device(collection_device))
    if collection_workers > 1 and torch.device(collection_device).type != "cpu":
        raise ValueError("parallel DAgger collection requires a CPU collection_device")
    if device is not None:
        if training_device is not None and torch.device(training_device) != torch.device(device):
            raise ValueError("device and training_device must match when both are supplied")
        training_device = device
    if training_device is None:
        training_device = "cuda" if torch.cuda.is_available() else "cpu"
    training_device = str(torch.device(training_device))
    highest_seed = (
        rounds * DAGGER_ROUND_SEED_STRIDE
        + 6 * 100_000
        + episodes_per_tier
        - 1
    )
    if highest_seed >= TRAINING_SEED_LIMIT:
        raise ValueError(
            f"DAgger seed schedule reaches {highest_seed}, outside the training namespace "
            f"0..{TRAINING_SEED_LIMIT - 1}"
        )
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f"DAgger initial checkpoint does not exist: {initial_checkpoint}")
    if not any(base_dataset.glob("tier-*-seed-*.npz")):
        raise FileNotFoundError(f"DAgger base dataset has no episode files: {base_dataset}")
    datasets = [base_dataset]
    for prior_round in range(1, start_round):
        prior_data = output / f"round-{prior_round}" / "data"
        if not any(prior_data.glob("tier-*-seed-*.npz")):
            raise FileNotFoundError(
                f"Cannot resume DAgger round {start_round}; prior round {prior_round} "
                f"dataset is missing or empty: {prior_data}"
            )
        datasets.append(prior_data)
    checkpoint = initial_checkpoint
    for round_number in range(start_round, rounds + 1):
        beta = max(0.0, beta_start * beta_decay ** (round_number - 1))
        round_data = output / f"round-{round_number}" / "data"
        collect_trajectories(
            round_data,
            tiers=range(1, 7),
            episodes_per_tier=episodes_per_tier,
            seed_start=round_number * DAGGER_ROUND_SEED_STRIDE,
            behavior_checkpoint=checkpoint,
            teacher_probability=beta,
            collection_device=collection_device,
            workers=collection_workers,
            overwrite=True,
        )
        datasets.append(round_data)
        round_model = output / f"round-{round_number}" / "model"
        checkpoint = train_behavior_clone(
            datasets,
            round_model,
            updates=updates_per_round,
            sequence_length=sequence_length,
            burn_in=burn_in,
            recurrent_size=recurrent_size,
            device=training_device,
            resume=False,
            init_checkpoint=checkpoint,
            latest_root_fraction=0.5,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
            uniform_window_fraction=uniform_window_fraction,
            validation_windows=validation_windows,
        )
    return checkpoint


def _parse_tiers(value: str) -> tuple[int, ...]:
    tiers = tuple(int(item) for item in value.split(","))
    if not tiers or any(tier not in range(1, 7) for tier in tiers):
        raise argparse.ArgumentTypeError("tiers must be a comma-separated subset of 1..6")
    return tiers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ghostline fair-teacher imitation and DAgger pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--tiers", type=_parse_tiers, default=tuple(range(1, 7)))
    collect.add_argument("--episodes-per-tier", type=int, default=100)
    collect.add_argument("--seed-start", type=int, default=0)
    collect.add_argument("--lesson", type=int, choices=range(0, 8), default=0)
    collect.add_argument("--behavior-checkpoint", type=Path)
    collect.add_argument("--teacher-probability", type=float, default=1.0)
    collect.add_argument(
        "--collection-device",
        "--device",
        dest="collection_device",
        default="cpu",
        help="behavior-policy rollout device; CPU is required with multiple workers",
    )
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--success-only", action="store_true")
    collect.add_argument("--overwrite", action="store_true")
    bc = commands.add_parser("bc")
    bc.add_argument("--dataset", type=Path, action="append", required=True)
    bc.add_argument("--output", type=Path, required=True)
    bc.add_argument("--updates", type=int, default=20_000)
    bc.add_argument("--batch-size", type=int, default=16)
    bc.add_argument("--sequence-length", type=int, default=DEFAULT_BC_SEQUENCE_LENGTH)
    bc.add_argument("--burn-in", type=int, default=DEFAULT_BC_BURN_IN)
    bc.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    bc.add_argument("--split-seed", type=int, default=0)
    bc.add_argument(
        "--uniform-window-fraction",
        type=float,
        default=DEFAULT_UNIFORM_WINDOW_FRACTION,
    )
    bc.add_argument("--validation-windows", type=int, default=DEFAULT_VALIDATION_WINDOWS)
    bc.add_argument("--learning-rate", type=float, default=2e-4)
    bc.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    bc.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    bc.add_argument("--init-checkpoint", type=Path)
    bc.add_argument("--no-resume", action="store_true")
    dagger = commands.add_parser("dagger")
    dagger.add_argument("--base-dataset", type=Path, required=True)
    dagger.add_argument("--initial-checkpoint", type=Path, required=True)
    dagger.add_argument("--output", type=Path, required=True)
    dagger.add_argument("--rounds", type=int, default=3, help="final numbered round to run")
    dagger.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="first numbered round to run; prior round datasets are included automatically",
    )
    dagger.add_argument("--episodes-per-tier", type=int, default=50)
    dagger.add_argument("--updates-per-round", type=int, default=5_000)
    dagger.add_argument("--beta-start", type=float, default=0.5)
    dagger.add_argument("--beta-decay", type=float, default=0.5)
    dagger.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    dagger.add_argument("--collection-device", default="cpu")
    dagger.add_argument(
        "--training-device",
        "--device",
        dest="training_device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    dagger.add_argument(
        "--collection-workers",
        type=int,
        default=DEFAULT_DAGGER_COLLECTION_WORKERS,
    )
    dagger.add_argument("--sequence-length", type=int, default=DEFAULT_BC_SEQUENCE_LENGTH)
    dagger.add_argument("--burn-in", type=int, default=DEFAULT_BC_BURN_IN)
    dagger.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    dagger.add_argument("--split-seed", type=int, default=0)
    dagger.add_argument(
        "--uniform-window-fraction",
        type=float,
        default=DEFAULT_UNIFORM_WINDOW_FRACTION,
    )
    dagger.add_argument("--validation-windows", type=int, default=DEFAULT_VALIDATION_WINDOWS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "collect":
        result = collect_trajectories(
            args.output,
            tiers=args.tiers,
            episodes_per_tier=args.episodes_per_tier,
            seed_start=args.seed_start,
            lesson=args.lesson,
            behavior_checkpoint=args.behavior_checkpoint,
            teacher_probability=args.teacher_probability,
            collection_device=args.collection_device,
            workers=args.workers,
            success_only=args.success_only,
            overwrite=args.overwrite,
        )
        print(json.dumps({**result.__dict__, "output": str(result.output), "behavior_success_rate": result.behavior_success_rate}, indent=2))
    elif args.command == "bc":
        print(
            train_behavior_clone(
                args.dataset,
                args.output,
                updates=args.updates,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                burn_in=args.burn_in,
                learning_rate=args.learning_rate,
                recurrent_size=args.recurrent_size,
                device=args.device,
                resume=not args.no_resume,
                init_checkpoint=args.init_checkpoint,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                uniform_window_fraction=args.uniform_window_fraction,
                validation_windows=args.validation_windows,
            )
        )
    else:
        print(
            run_dagger(
                args.base_dataset,
                args.output,
                args.initial_checkpoint,
                rounds=args.rounds,
                start_round=args.start_round,
                episodes_per_tier=args.episodes_per_tier,
                updates_per_round=args.updates_per_round,
                beta_start=args.beta_start,
                beta_decay=args.beta_decay,
                recurrent_size=args.recurrent_size,
                collection_device=args.collection_device,
                training_device=args.training_device,
                collection_workers=args.collection_workers,
                sequence_length=args.sequence_length,
                burn_in=args.burn_in,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                uniform_window_fraction=args.uniform_window_fraction,
                validation_windows=args.validation_windows,
            )
        )


if __name__ == "__main__":
    main()
