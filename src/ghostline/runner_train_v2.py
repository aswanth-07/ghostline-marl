"""Recurrent PPO training for the in-development Ghostline runner v2.

This module is intentionally independent from :mod:`ghostline.torchrl_train`.
The latter is part of the published single-agent v1 lineage and its observation
and checkpoint contracts must remain frozen.

The implementation here prioritises training correctness over convenience:

* the complete 288-action legality mask is sampled and re-used during PPO;
* recurrent minibatches preserve time and reset the GRU at episode boundaries;
* every worker draws from a deterministic, non-overlapping training schedule;
* in-progress episodes are restored by deterministic action replay;
* checkpoints fail closed on environment, model, trainer and argument drift;
* both global and local NumPy RNG state and Torch CPU/CUDA RNG state are saved;
* validation windows are disjoint and acceptance needs two consecutive passes;
* non-finite losses or gradients abort before an optimizer step/checkpoint.

``main(args=None)`` is deliberately exposed even before the shared Ghostline CLI
is wired to this development track.  That keeps smoke runs and long campaigns
available without changing the published player command surface.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ghostline.curriculum import ACCEPTANCE_THRESHOLDS, PROMOTION_THRESHOLDS
from ghostline.env_v2 import GhostlineEnvV2
from ghostline.model_v2 import (
    OBSERVATION_CONTRACT_V2,
    RunnerPolicyV2,
    initialize_runner_v2_from_published_v1,
    load_runner_v2,
    multi_agent_environment_fingerprint,
    runner_model_fingerprint,
)
from ghostline.seeds import (
    FINAL_TEST_SEED_START,
    TRAINING_SEED_END,
    TRAINING_SEED_START,
    VALIDATION_SEED_END,
    VALIDATION_SEED_START,
    VALIDATION_TIER_STRIDE,
    validation_seed,
)
from ghostline.types_v2 import (
    RUNNER_ACTION_COUNT_V2,
    ContractDirective,
)


TRAINER_CONTRACT_V2 = "ghostline-runner-recurrent-ppo-v2.3"
EXPERIMENT_MANIFEST_CONTRACT = "ghostline-runner-v2-experiment-manifest-v1"
CHECKPOINT_VERSION = 2
OBSERVATION_KEYS_V2 = (
    "ego",
    "objective",
    "directive",
    "field",
    "field_targets",
    "field_target_mask",
    "local_grid",
    "targets",
    "target_mask",
    "entities",
    "entity_mask",
    "rays",
    "action_mask",
)
ALL_TIERS = (1, 2, 3, 4, 5, 6)
ALL_DIRECTIVES = (
    ContractDirective.STANDARD,
    ContractDirective.GHOST,
    ContractDirective.SPEED,
    ContractDirective.GREED,
)
GHOST_TRAINING_STAGE_COUNTS: dict[int, tuple[int, int] | None] = {
    # Zero is the real environment. The other stages are training-only
    # stepping stones between camera-only tier 2 and full tier 3.
    0: None,
    1: (1, 0),
    2: (1, 1),
    3: (2, 1),
}


def apply_training_ghost_security_stage(
    environment: GhostlineEnvV2,
    *,
    stage: int,
    tier: int,
    directive: ContractDirective | str | int,
    observation: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Apply a declared training-only Ghost security roster.

    The public environment and every release evaluation keep ``stage=0``.
    Easier rosters are constructed only after a normal deterministic reset, so
    geometry, objectives, observations, rewards, and action semantics remain
    the real v2 contract. The actor still receives only player-equivalent
    observations.
    """

    stage = int(stage)
    if stage not in GHOST_TRAINING_STAGE_COUNTS:
        raise ValueError("ghost training stage must lie in 0..3")
    parsed_directive = ContractDirective.parse(directive)
    roster = GHOST_TRAINING_STAGE_COUNTS[stage]
    if roster is None or parsed_directive != ContractDirective.GHOST or int(tier) < 3:
        if observation is not None:
            return {
                key: np.asarray(value)
                for key, value in observation.items()
            }
        return environment._observation()

    guard_count, camera_count = roster
    sim = environment.sim
    sim.level.guards = list(sim.level.guards[:guard_count])
    sim.level.cameras = list(sim.level.cameras[:camera_count])
    sim.level.response_drones = False
    sim.drones = []

    active_guard_ids = {int(guard.guard_id) for guard in sim.level.guards}
    active_camera_ids = {int(camera.camera_id) for camera in sim.level.cameras}
    sim.operative_states = {
        guard_id: state
        for guard_id, state in sim.operative_states.items()
        if int(guard_id) in active_guard_ids
    }
    sim.sensor_charges = {
        guard_id: charges
        for guard_id, charges in sim.sensor_charges.items()
        if int(guard_id) in active_guard_ids
    }
    sim._pending_security_orders = {
        guard_id: order
        for guard_id, order in sim._pending_security_orders.items()
        if int(guard_id) in active_guard_ids
    }
    sim.security_intel = {
        key: value
        for key, value in sim.security_intel.items()
        if (
            (key[0] == "guard" and int(key[1]) in active_guard_ids)
            or (key[0] == "camera" and int(key[1]) in active_camera_ids)
        )
    }
    sim._guard_waypoints.clear()
    sim._drone_waypoints.clear()

    # Frozen learned security is not used by the first curriculum campaign,
    # but keeping this helper correct prevents stale recurrent slots if a
    # declared future ablation combines the two.
    controller = getattr(environment, "security_controller", None)
    if controller is not None:
        controller.reset(sim)
        controller.update(force=True)
    return environment._observation()


def _normalised_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_trainer_fingerprint(path: Path | None = None) -> str:
    """Fingerprint PPO semantics so a changed trainer cannot resume silently."""

    source = Path(__file__) if path is None else Path(path)
    return hashlib.sha256(
        (
            TRAINER_CONTRACT_V2
            + ":"
            + _normalised_file_hash(source)
        ).encode("utf-8")
    ).hexdigest()


def runtime_fingerprint() -> dict[str, str]:
    """Versions whose state-dict/RNG semantics must agree on strict resume."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
    }


def _git_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Return auditable source provenance without requiring a Git checkout."""

    repository = (
        Path(__file__).resolve().parents[2]
        if root is None
        else Path(root).resolve()
    )

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--porcelain=v1", "--untracked-files=normal")
    except (FileNotFoundError, subprocess.SubprocessError):
        return {
            "available": False,
            "repository": str(repository),
            "commit": None,
            "branch": None,
            "dirty": None,
        }
    return {
        "available": True,
        "repository": str(repository),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
    }


def _hardware_snapshot(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "selected_device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": (
            torch.cuda.device_count() if cuda_available else 0
        ),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }


def build_experiment_manifest(
    *,
    args: argparse.Namespace,
    config: RunnerPPOConfig,
    initialization: Mapping[str, Any],
    device: torch.device,
    status: str,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Freeze every input needed to audit or reproduce a runner campaign."""

    security_opponents = [
        {
            "path": str(Path(path).resolve()),
            "sha256": digest,
        }
        for path, digest in zip(
            config.security_opponent_paths,
            config.security_opponent_sha256,
            strict=True,
        )
    ]
    resume_record: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_record = {
            "path": str(resume_checkpoint.resolve()),
            "sha256": _sha256(resume_checkpoint),
        }
    return {
        "manifest_contract": EXPERIMENT_MANIFEST_CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "public_environment": OBSERVATION_CONTRACT_V2,
        "checkpoint_contract": _checkpoint_contract(config),
        "initialization": dict(initialization),
        "resume_checkpoint": resume_record,
        "security_opponents": security_opponents,
        "seed_namespaces": {
            "training": {
                "start": TRAINING_SEED_START,
                "end": TRAINING_SEED_END,
                "configured_start": config.training_seed_start,
            },
            "validation": {
                "start": VALIDATION_SEED_START,
                "end": VALIDATION_SEED_END,
                "configured_cursor": config.initial_validation_cursor,
                "tier_stride": VALIDATION_TIER_STRIDE,
            },
            "final_test": {
                "reserved_start": FINAL_TEST_SEED_START,
                "not_consumed_by_training": True,
            },
        },
        "budget": {
            "max_updates": int(getattr(args, "max_updates", 0)),
            "max_decisions": int(getattr(args, "max_decisions", 0)),
            "seconds": float(getattr(args, "seconds", 0.0)),
            "stop_on_acceptance": bool(
                getattr(args, "stop_on_acceptance", True)
            ),
        },
        "hardware": _hardware_snapshot(device),
        "source": _git_snapshot(),
    }


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    """Durably replace a small JSON record without exposing partial content."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stable_mix(value: int) -> int:
    """SplitMix64 finalizer used for deterministic schedule choices."""

    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def parse_tiers(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in value)
    if not values or len(set(values)) != len(values) or any(tier not in ALL_TIERS for tier in values):
        raise ValueError("tiers must be a non-empty unique subset of 1,2,3,4,5,6")
    return values


def parse_directives(
    value: str | Iterable[ContractDirective | str | int],
) -> tuple[ContractDirective, ...]:
    if isinstance(value, str):
        raw: Iterable[ContractDirective | str | int] = (
            item.strip() for item in value.split(",") if item.strip()
        )
    else:
        raw = value
    values = tuple(ContractDirective.parse(item) for item in raw)
    if not values or len(set(values)) != len(values):
        raise ValueError("directives must be a non-empty unique list")
    return values


@dataclass(frozen=True)
class RunnerPPOConfig:
    """Semantic PPO configuration included in the strict resume contract."""

    seed: int = 7
    envs: int = 8
    rollout: int = 512
    epochs: int = 4
    minibatch_envs: int = 2
    recurrent_size: int = 384
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    reward_scale: float = 0.05
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    objective_aux_coefficient: float = 0.15
    danger_aux_coefficient: float = 0.10
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    training_seed_start: int = TRAINING_SEED_START
    tiers: tuple[int, ...] = ALL_TIERS
    directives: tuple[int, ...] = tuple(int(item) for item in ALL_DIRECTIVES)
    ghost_directive_fraction: float = 0.25
    ghost_training_stage: int = 0
    adaptive_curriculum: bool = True
    initial_curriculum_tier: int = 1
    async_envs: bool = True
    validation_interval: int = 100
    validation_episodes: int = 25
    validation_batch_size: int = 16
    initial_validation_cursor: int = 0
    checkpoint_interval: int = 1
    security_opponent_paths: tuple[str, ...] = ()
    security_opponent_sha256: tuple[str, ...] = ()
    security_pool_salt: int = 0

    def validate(self) -> None:
        if self.envs <= 0:
            raise ValueError("envs must be positive")
        if self.rollout <= 0:
            raise ValueError("rollout must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 1 <= self.minibatch_envs <= self.envs:
            raise ValueError("minibatch_envs must lie in 1..envs")
        if self.recurrent_size not in (256, 384, 512):
            raise ValueError("recurrent_size must be 256, 384 or 512")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.reward_scale <= 1.0:
            raise ValueError("reward_scale must lie in (0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if not 0.0 < self.value_clip_ratio < 1.0:
            raise ValueError("value_clip_ratio must lie in (0, 1)")
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if (
            self.objective_aux_coefficient < 0.0
            or self.danger_aux_coefficient < 0.0
        ):
            raise ValueError("auxiliary coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_kl < 0.0:
            raise ValueError("target_kl must be non-negative")
        parse_tiers(self.tiers)
        parse_directives(self.directives)
        if not 0.0 <= self.ghost_directive_fraction <= 1.0:
            raise ValueError("ghost directive fraction must lie in [0, 1]")
        if (
            self.ghost_directive_fraction != 0.0
            and int(ContractDirective.GHOST) not in self.directives
        ):
            raise ValueError(
                "a positive ghost directive fraction requires the ghost directive"
            )
        if self.ghost_training_stage not in GHOST_TRAINING_STAGE_COUNTS:
            raise ValueError("ghost_training_stage must lie in 0..3")
        if self.ghost_training_stage:
            if self.adaptive_curriculum:
                raise ValueError(
                    "training-only Ghost security stages require --no-curriculum"
                )
            if self.directives != (int(ContractDirective.GHOST),):
                raise ValueError(
                    "training-only Ghost security stages require --directives ghost"
                )
            if self.security_opponent_paths:
                raise ValueError(
                    "training-only Ghost security stages require scripted security"
                )
        if self.adaptive_curriculum:
            if self.tiers != ALL_TIERS:
                raise ValueError(
                    "adaptive curriculum requires the complete ordered six-tier distribution"
                )
            if self.initial_curriculum_tier not in ALL_TIERS:
                raise ValueError("initial_curriculum_tier must lie in 1..6")
        require_training_schedule(
            start=self.training_seed_start,
            env_count=self.envs,
        )
        if self.validation_interval < 0:
            raise ValueError("validation_interval must be non-negative")
        if self.validation_interval and self.validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive when validation is enabled")
        if self.validation_interval and self.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be positive when validation is enabled")
        if self.validation_interval:
            require_validation_window(
                self.initial_validation_cursor,
                self.validation_episodes,
            )
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if len(self.security_opponent_paths) != len(
            self.security_opponent_sha256
        ):
            raise ValueError("security opponent paths and hashes must align")
        if len(set(self.security_opponent_sha256)) != len(
            self.security_opponent_sha256
        ):
            raise ValueError("security opponent checkpoints must be unique")
        for raw_path, expected_hash in zip(
            self.security_opponent_paths,
            self.security_opponent_sha256,
            strict=True,
        ):
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"security opponent checkpoint is missing: {path}"
                )
            if _sha256(path) != expected_hash:
                raise RuntimeError(
                    f"security opponent checkpoint bytes changed: {path}"
                )

    def contract(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tiers"] = list(self.tiers)
        payload["directives"] = list(self.directives)
        return payload


def require_training_schedule(*, start: int, env_count: int) -> None:
    if env_count <= 0:
        raise ValueError("env_count must be positive")
    if not TRAINING_SEED_START <= int(start) <= TRAINING_SEED_END:
        raise ValueError(
            f"training seed start must lie in {TRAINING_SEED_START}..{TRAINING_SEED_END}"
        )
    if int(start) + env_count - 1 > TRAINING_SEED_END:
        raise ValueError("initial vector seeds leave the training namespace")


def require_validation_window(cursor: int, episodes: int) -> None:
    if episodes <= 0:
        raise ValueError("validation episodes must be positive")
    if cursor < 0 or cursor + episodes > VALIDATION_TIER_STRIDE:
        raise ValueError(
            "validation window leaves or exhausts its per-tier namespace"
        )


class ScheduledRunnerEnv(gym.Wrapper):
    """Runner env with a disjoint per-worker seed stream and replay snapshot."""

    def __init__(
        self,
        *,
        rank: int,
        env_count: int,
        training_seed_start: int,
        tiers: Sequence[int],
        directives: Sequence[int],
        schedule_salt: int,
        adaptive_curriculum: bool,
        initial_curriculum_tier: int,
        ghost_directive_fraction: float = 0.25,
        ghost_training_stage: int = 0,
        security_opponent_paths: Sequence[str] = (),
        security_opponent_sha256: Sequence[str] = (),
        security_pool_salt: int = 0,
        reward_gamma: float = 0.999,
    ):
        if not 0 <= rank < env_count:
            raise ValueError("rank must lie in 0..env_count-1")
        require_training_schedule(start=training_seed_start, env_count=env_count)
        self.rank = int(rank)
        self.env_count = int(env_count)
        self.training_seed_start = int(training_seed_start)
        self.tiers = parse_tiers(tiers)
        self.directives = parse_directives(directives)
        self.ghost_directive_fraction = float(ghost_directive_fraction)
        self.ghost_training_stage = int(ghost_training_stage)
        if self.ghost_training_stage not in GHOST_TRAINING_STAGE_COUNTS:
            raise ValueError("ghost_training_stage must lie in 0..3")
        self.schedule_salt = int(schedule_salt)
        self.adaptive_curriculum = bool(adaptive_curriculum)
        self.curriculum_tier = int(initial_curriculum_tier)
        self.episode_curriculum_tier = int(initial_curriculum_tier)
        self.security_opponent_paths = tuple(
            str(Path(path).resolve())
            for path in security_opponent_paths
        )
        self.security_opponent_sha256 = tuple(security_opponent_sha256)
        if len(self.security_opponent_paths) != len(
            self.security_opponent_sha256
        ):
            raise ValueError("security opponent paths and hashes must align")
        for path, expected_hash in zip(
            self.security_opponent_paths,
            self.security_opponent_sha256,
            strict=True,
        ):
            if _sha256(Path(path)) != expected_hash:
                raise RuntimeError(
                    f"security opponent checkpoint changed before worker start: {path}"
                )
        self.security_pool_salt = int(security_pool_salt)
        self.reward_gamma = float(reward_gamma)
        if not 0.0 < self.reward_gamma <= 1.0:
            raise ValueError("reward_gamma must lie in (0, 1]")
        self.next_seed = self.training_seed_start + self.rank
        self.current_seed: int | None = None
        self.current_tier: int | None = None
        self.current_directive: ContractDirective | None = None
        self.action_history: list[int] = []
        self.last_observation: dict[str, np.ndarray] | None = None
        initial_seed = min(self.next_seed, TRAINING_SEED_END)
        if self.security_opponent_paths:
            from ghostline.security_opponents import FrozenSecurityRunnerEnvV2

            environment: GhostlineEnvV2 = FrozenSecurityRunnerEnvV2(
                security_checkpoints=tuple(
                    Path(path) for path in self.security_opponent_paths
                ),
                security_pool_salt=self.security_pool_salt,
                seed=initial_seed,
                tier=self.tiers[0],
                reward_gamma=self.reward_gamma,
            )
        else:
            environment = GhostlineEnvV2(
                seed=initial_seed,
                tier=self.tiers[0],
                reward_gamma=self.reward_gamma,
            )
        super().__init__(environment)

    def _schedule(
        self,
        seed: int,
        *,
        curriculum_tier: int | None = None,
    ) -> tuple[int, ContractDirective]:
        curriculum_tier = (
            self.curriculum_tier
            if curriculum_tier is None
            else int(curriculum_tier)
        )
        tier_draw = _stable_mix(seed ^ self.schedule_salt)
        if not self.adaptive_curriculum:
            tier = self.tiers[tier_draw % len(self.tiers)]
        elif curriculum_tier == 1:
            tier = 1
        else:
            # At tiers 2-5 replay is 30%; at tier 6 the full distribution is
            # a deliberate 50/50 split, matching the versioned project plan.
            current_weight = 0.50 if curriculum_tier == 6 else 0.70
            unit_draw = tier_draw / float(1 << 64)
            if unit_draw < current_weight:
                tier = curriculum_tier
            else:
                replay_draw = _stable_mix(tier_draw ^ 0x94D049BB133111EB)
                tier = 1 + replay_draw % (curriculum_tier - 1)
        directive_draw = _stable_mix(
            seed ^ self.schedule_salt ^ 0xD1B54A32D192ED03
        )
        ghost = ContractDirective.GHOST
        non_ghost = tuple(
            directive for directive in self.directives if directive != ghost
        )
        if (
            ghost in self.directives
            and non_ghost
            and self.ghost_directive_fraction != 1.0 / len(self.directives)
        ):
            unit_draw = directive_draw / float(1 << 64)
            if unit_draw < self.ghost_directive_fraction:
                directive = ghost
            else:
                replay_draw = _stable_mix(
                    directive_draw ^ 0xA24BAED4963EE407
                )
                directive = non_ghost[replay_draw % len(non_ghost)]
        else:
            directive = self.directives[directive_draw % len(self.directives)]
        return tier, directive

    def set_curriculum_tier(self, tier: int) -> None:
        tier = int(tier)
        if tier not in ALL_TIERS:
            raise ValueError("curriculum tier must lie in 1..6")
        if tier < self.curriculum_tier:
            raise RuntimeError("runner curriculum cannot move backwards")
        self.curriculum_tier = tier

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del seed  # The versioned scheduler owns all training seeds.
        if self.next_seed > TRAINING_SEED_END:
            raise RuntimeError(
                "runner training seed namespace exhausted; start a new declared namespace"
            )
        episode_seed = self.next_seed
        self.next_seed += self.env_count
        tier, directive = self._schedule(episode_seed)
        requested = dict(options or {})
        requested.update({"tier": tier, "directive": directive})
        observation, info = self.env.reset(seed=episode_seed, options=requested)
        observation = apply_training_ghost_security_stage(
            self.env,
            stage=self.ghost_training_stage,
            tier=tier,
            directive=directive,
            observation=observation,
        )
        if self.ghost_training_stage:
            info = dict(info)
            info["training_ghost_security_stage"] = self.ghost_training_stage
        self.current_seed = episode_seed
        self.current_tier = tier
        self.current_directive = directive
        self.episode_curriculum_tier = self.curriculum_tier
        self.action_history = []
        self.last_observation = observation
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_value = int(action)
        observation, reward, terminated, truncated, info = self.env.step(action_value)
        self.action_history.append(action_value)
        self.last_observation = observation
        return observation, reward, terminated, truncated, info

    def checkpoint_state(self) -> dict[str, Any]:
        if self.current_seed is None or self.current_tier is None or self.current_directive is None:
            raise RuntimeError("scheduled environment has not been reset")
        return {
            "rank": self.rank,
            "next_seed": self.next_seed,
            "current_seed": self.current_seed,
            "current_tier": self.current_tier,
            "current_directive": int(self.current_directive),
            "curriculum_tier": self.curriculum_tier,
            "episode_curriculum_tier": self.episode_curriculum_tier,
            "action_history": list(self.action_history),
            "observation_digest": observation_digest(self.last_observation),
        }

    def restore_state(
        self,
        states: Sequence[Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        """Restore this worker by replaying its deterministic current episode."""

        state = dict(states[self.rank])
        if int(state.get("rank", -1)) != self.rank:
            raise RuntimeError("vector checkpoint rank ordering changed")
        current_seed = int(state["current_seed"])
        next_seed = int(state["next_seed"])
        if not TRAINING_SEED_START <= current_seed <= TRAINING_SEED_END:
            raise RuntimeError("checkpoint current seed left the training namespace")
        if next_seed != current_seed + self.env_count:
            raise RuntimeError("checkpoint worker seed stride changed")
        tier = int(state["current_tier"])
        directive = ContractDirective.parse(state["current_directive"])
        curriculum_tier = int(state["curriculum_tier"])
        episode_curriculum_tier = int(state["episode_curriculum_tier"])
        expected_tier, expected_directive = self._schedule(
            current_seed,
            curriculum_tier=episode_curriculum_tier,
        )
        if tier != expected_tier or directive != expected_directive:
            raise RuntimeError("checkpoint episode schedule no longer matches configuration")
        observation, _ = self.env.reset(
            seed=current_seed,
            options={"tier": tier, "directive": directive},
        )
        observation = apply_training_ghost_security_stage(
            self.env,
            stage=self.ghost_training_stage,
            tier=tier,
            directive=directive,
            observation=observation,
        )
        actions = [int(value) for value in state.get("action_history", ())]
        for index, action in enumerate(actions):
            observation, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"checkpoint replay ended at action {index}; snapshots must be post-autoreset"
                )
        if observation_digest(observation) != state.get("observation_digest"):
            raise RuntimeError("deterministic environment replay changed observation bytes")
        self.current_seed = current_seed
        self.next_seed = next_seed
        self.current_tier = tier
        self.current_directive = directive
        self.curriculum_tier = curriculum_tier
        self.episode_curriculum_tier = episode_curriculum_tier
        self.action_history = actions
        self.last_observation = observation
        return observation


@dataclass(frozen=True)
class RunnerEnvFactory:
    """Pickle-safe worker factory for Windows ``spawn``."""

    rank: int
    env_count: int
    training_seed_start: int
    tiers: tuple[int, ...]
    directives: tuple[int, ...]
    ghost_directive_fraction: float
    ghost_training_stage: int
    schedule_salt: int
    adaptive_curriculum: bool
    initial_curriculum_tier: int
    security_opponent_paths: tuple[str, ...]
    security_opponent_sha256: tuple[str, ...]
    security_pool_salt: int
    reward_gamma: float

    def __call__(self) -> ScheduledRunnerEnv:
        return ScheduledRunnerEnv(
            rank=self.rank,
            env_count=self.env_count,
            training_seed_start=self.training_seed_start,
            tiers=self.tiers,
            directives=self.directives,
            ghost_directive_fraction=self.ghost_directive_fraction,
            ghost_training_stage=self.ghost_training_stage,
            schedule_salt=self.schedule_salt,
            adaptive_curriculum=self.adaptive_curriculum,
            initial_curriculum_tier=self.initial_curriculum_tier,
            security_opponent_paths=self.security_opponent_paths,
            security_opponent_sha256=self.security_opponent_sha256,
            security_pool_salt=self.security_pool_salt,
            reward_gamma=self.reward_gamma,
        )


def make_runner_vector_env(
    config: RunnerPPOConfig,
) -> gym.vector.VectorEnv:
    """Create same-step vector workers with deterministic episode schedules."""

    config.validate()
    factories = [
        RunnerEnvFactory(
            rank=rank,
            env_count=config.envs,
            training_seed_start=config.training_seed_start,
            tiers=config.tiers,
            directives=config.directives,
            ghost_directive_fraction=config.ghost_directive_fraction,
            ghost_training_stage=config.ghost_training_stage,
            schedule_salt=config.seed,
            adaptive_curriculum=config.adaptive_curriculum,
            initial_curriculum_tier=config.initial_curriculum_tier,
            security_opponent_paths=config.security_opponent_paths,
            security_opponent_sha256=config.security_opponent_sha256,
            security_pool_salt=config.security_pool_salt,
            reward_gamma=config.gamma,
        )
        for rank in range(config.envs)
    ]
    vector_type = gym.vector.AsyncVectorEnv if config.async_envs else gym.vector.SyncVectorEnv
    return vector_type(
        factories,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def observation_digest(
    observation: Mapping[str, np.ndarray] | None,
) -> str:
    if observation is None:
        raise RuntimeError("missing current observation")
    digest = hashlib.sha256()
    if tuple(observation) != OBSERVATION_KEYS_V2:
        # Dict insertion order is not a semantic contract, so hash canonically.
        keys = OBSERVATION_KEYS_V2
    else:
        keys = tuple(observation)
    for key in keys:
        if key not in observation:
            raise RuntimeError(f"observation omitted {key!r}")
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _stack_observations(
    observations: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("cannot stack an empty observation sequence")
    return {
        key: np.stack([np.asarray(observation[key]) for observation in observations])
        for key in OBSERVATION_KEYS_V2
    }


def _tensor_observation(
    observation: Mapping[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(observation[key], device=device)
        for key in OBSERVATION_KEYS_V2
    }


def _assert_action_contract(
    observation: Mapping[str, np.ndarray | torch.Tensor],
) -> None:
    mask = observation["action_mask"]
    if mask.shape[-1] != RUNNER_ACTION_COUNT_V2:
        raise RuntimeError(
            f"runner mask changed: expected {RUNNER_ACTION_COUNT_V2}, got {mask.shape[-1]}"
        )
    if isinstance(mask, torch.Tensor):
        legal_counts = (mask > 0).sum(dim=-1)
        valid_values = torch.logical_or(mask == 0, mask == 1).all()
        if not bool(valid_values) or bool((legal_counts <= 0).any()):
            raise RuntimeError("runner action mask is non-binary or has an empty row")
    else:
        values = np.asarray(mask)
        if not np.logical_or(values == 0, values == 1).all():
            raise RuntimeError("runner action mask must be binary")
        if np.any(np.sum(values > 0, axis=-1) <= 0):
            raise RuntimeError("runner action mask has an empty row")


def _zero_hidden_at_starts(
    hidden: torch.Tensor,
    starts: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    reset = torch.as_tensor(starts, dtype=torch.bool, device=hidden.device)
    if reset.ndim != 1 or reset.shape[0] != hidden.shape[1]:
        raise ValueError("episode-start mask must match the hidden-state batch")
    if not bool(reset.any()):
        return hidden
    result = hidden.clone()
    result[:, reset, :] = 0.0
    return result


@dataclass
class RunnerSequenceOutputs:
    logits: torch.Tensor
    values: torch.Tensor
    objective_bearing: torch.Tensor
    danger: torch.Tensor
    next_hidden: torch.Tensor


def public_auxiliary_labels(
    observation: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build labels exclusively from the player-visible observation contract."""

    objective = observation["objective"].float()
    waypoint = objective[..., 4:6]
    goal = objective[..., 1:3]
    waypoint_norm = torch.linalg.vector_norm(waypoint, dim=-1, keepdim=True)
    direction = torch.where(waypoint_norm > 0.04, waypoint, goal)
    magnitude = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    bearing = torch.where(
        magnitude > 1e-6,
        direction / magnitude.clamp_min(1e-6),
        torch.zeros_like(direction),
    )
    # Ray channel one is the public danger distance already shown to the
    # player-equivalent controller. Projectile pressure remains a separate
    # fourth channel and is not converted into a privileged binary label.
    danger = observation["rays"].float()[..., 1].amax(dim=-1)
    return bearing, danger.clamp(0.0, 1.0)


def runner_sequence_outputs(
    policy: RunnerPolicyV2,
    observation: Mapping[str, torch.Tensor],
    hidden: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> RunnerSequenceOutputs:
    """Decode PPO and auxiliary heads from one recurrent latent traversal."""

    time_steps, batch = observation["ego"].shape[:2]
    flattened = {
        key: value.flatten(0, 1)
        for key, value in observation.items()
    }
    encoded = policy.encode(flattened).reshape(
        time_steps,
        batch,
        -1,
    ).transpose(0, 1)
    if reset_mask is None:
        sequence, next_hidden = policy.core(encoded, hidden)
    else:
        outputs: list[torch.Tensor] = []
        next_hidden = hidden
        boundaries = torch.nonzero(
            reset_mask.to(encoded.device).bool().any(dim=1),
            as_tuple=False,
        ).flatten().tolist()
        starts = sorted({0, *boundaries, time_steps})
        for start, end in zip(starts[:-1], starts[1:], strict=True):
            reset = reset_mask[start].to(encoded.device).bool()
            if bool(reset.any()) and next_hidden is not None:
                next_hidden = next_hidden.clone()
                next_hidden[:, reset, :] = 0.0
            output, next_hidden = policy.core(
                encoded[:, start:end],
                next_hidden,
            )
            outputs.append(output)
        sequence = torch.cat(outputs, dim=1)
    latent = sequence.transpose(0, 1)
    logits = policy.action_logits(latent, observation["action_mask"])
    values = policy.value_head(
        policy.value_decoder(latent)
    ).squeeze(-1)
    objective_bearing = torch.tanh(policy.objective_head(latent))
    danger = torch.sigmoid(policy.danger_head(latent).squeeze(-1))
    return RunnerSequenceOutputs(
        logits=logits,
        values=values,
        objective_bearing=objective_bearing,
        danger=danger,
        next_hidden=next_hidden,
    )


@dataclass
class RunnerRollout:
    observations: dict[str, np.ndarray]
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    episode_starts: np.ndarray
    initial_hidden: torch.Tensor
    advantages: np.ndarray
    returns: np.ndarray
    next_observation: dict[str, np.ndarray]
    next_hidden: torch.Tensor
    next_episode_starts: np.ndarray
    completed_successes: list[float]

    @property
    def time_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def env_count(self) -> int:
        return int(self.actions.shape[1])


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with semantic failures/timeouts as terminal transitions."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values and dones must have the same [time, env] shape")
    if rewards.ndim != 2 or bootstrap_values.shape != rewards.shape[1:]:
        raise ValueError("bootstrap_values must have shape [env]")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = bootstrap_values
    for index in range(rewards.shape[0] - 1, -1, -1):
        continuation = 1.0 - dones[index].astype(np.float32)
        delta = rewards[index] + gamma * next_values * continuation - values[index]
        last_advantage = (
            delta
            + gamma * gae_lambda * continuation * last_advantage
        )
        advantages[index] = last_advantage
        next_values = values[index]
    return advantages, advantages + values


def _completed_successes(infos: Mapping[str, Any], done: np.ndarray) -> list[float]:
    final_info = infos.get("final_info")
    final_mask = np.asarray(infos.get("_final_info", done), dtype=bool)
    results: list[float] = []
    if isinstance(final_info, Mapping):
        success = np.asarray(
            final_info.get("is_success", np.zeros_like(done, dtype=bool))
        )
        success_mask = np.asarray(
            final_info.get("_is_success", final_mask),
            dtype=bool,
        )
        for index, ended in enumerate(done):
            if ended and final_mask[index] and success_mask[index]:
                results.append(float(success[index]))
        return results
    if final_info is None:
        return results
    for index, ended in enumerate(done):
        if not ended or not final_mask[index]:
            continue
        item = final_info[index]
        if item is not None:
            results.append(float(item.get("is_success", False)))
    return results


@torch.no_grad()
def collect_rollout(
    *,
    policy: RunnerPolicyV2,
    vector_env: gym.vector.VectorEnv,
    observation: dict[str, np.ndarray],
    hidden: torch.Tensor,
    episode_starts: np.ndarray,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    reward_scale: float = 1.0,
) -> RunnerRollout:
    """Collect one vector rollout while preserving exact GRU reset semantics."""

    policy.eval()
    observation_buffer = {key: [] for key in OBSERVATION_KEYS_V2}
    actions: list[np.ndarray] = []
    log_probabilities: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    successes: list[float] = []

    hidden = _zero_hidden_at_starts(hidden, episode_starts)
    initial_hidden = hidden.detach().cpu().clone()
    current_observation = observation
    current_starts = np.asarray(episode_starts, dtype=bool)

    for _ in range(rollout_steps):
        _assert_action_contract(current_observation)
        hidden = _zero_hidden_at_starts(hidden, current_starts)
        tensors = _tensor_observation(current_observation, device)
        logits, value, next_hidden = policy(tensors, hidden)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        legal = tensors["action_mask"].gather(1, action[:, None]).squeeze(1) > 0
        if not bool(legal.all()):
            raise RuntimeError("masked policy sampled an illegal runner action")
        log_probability = distribution.log_prob(action)
        next_observation, reward, terminated, truncated, infos = vector_env.step(
            action.cpu().numpy()
        )
        done = np.logical_or(terminated, truncated)

        for key in OBSERVATION_KEYS_V2:
            observation_buffer[key].append(np.asarray(current_observation[key]).copy())
        actions.append(action.cpu().numpy())
        log_probabilities.append(log_probability.cpu().numpy())
        values.append(value.cpu().numpy())
        rewards.append(np.asarray(reward, dtype=np.float32))
        dones.append(done)
        starts.append(current_starts.copy())
        successes.extend(_completed_successes(infos, done))

        current_observation = {
            key: np.asarray(next_observation[key]) for key in OBSERVATION_KEYS_V2
        }
        hidden = next_hidden
        current_starts = done

    bootstrap_hidden = _zero_hidden_at_starts(hidden, current_starts)
    bootstrap_tensors = _tensor_observation(current_observation, device)
    _, bootstrap_value, _ = policy(bootstrap_tensors, bootstrap_hidden)
    values_array = np.stack(values).astype(np.float32)
    rewards_array = np.stack(rewards).astype(np.float32)
    dones_array = np.stack(dones).astype(bool)
    advantages, returns = compute_gae(
        rewards_array * float(reward_scale),
        values_array,
        dones_array,
        bootstrap_value.cpu().numpy(),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return RunnerRollout(
        observations={
            key: np.stack(buffer) for key, buffer in observation_buffer.items()
        },
        actions=np.stack(actions).astype(np.int64),
        old_log_probabilities=np.stack(log_probabilities).astype(np.float32),
        old_values=values_array,
        rewards=rewards_array,
        dones=dones_array,
        episode_starts=np.stack(starts).astype(bool),
        initial_hidden=initial_hidden,
        advantages=advantages,
        returns=returns,
        next_observation=current_observation,
        next_hidden=hidden.detach(),
        next_episode_starts=current_starts,
        completed_successes=successes,
    )


@dataclass(frozen=True)
class PPODiagnostics:
    policy_loss: float
    value_loss: float
    entropy: float
    objective_aux_loss: float
    danger_aux_loss: float
    weighted_auxiliary_loss: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    explained_variance: float
    epochs_completed: int
    early_stopped: bool
    samples: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {name}; optimizer step aborted")


def _normalise_advantages(advantages: np.ndarray) -> np.ndarray:
    values = np.asarray(advantages, dtype=np.float32)
    mean = float(values.mean())
    standard_deviation = float(values.std())
    return (values - mean) / max(standard_deviation, 1e-8)


def evaluate_rollout_log_probabilities(
    policy: RunnerPolicyV2,
    rollout: RunnerRollout,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-evaluate a complete rollout without breaking recurrent history."""

    observations = {
        key: torch.as_tensor(value, device=device)
        for key, value in rollout.observations.items()
    }
    _assert_action_contract(observations)
    outputs = runner_sequence_outputs(
        policy,
        observations,
        rollout.initial_hidden.to(device),
        torch.as_tensor(rollout.episode_starts, device=device),
    )
    distribution = torch.distributions.Categorical(logits=outputs.logits)
    actions = torch.as_tensor(rollout.actions, device=device)
    return (
        distribution.log_prob(actions),
        outputs.values,
        distribution.entropy(),
    )


def ppo_update(
    *,
    policy: RunnerPolicyV2,
    optimizer: torch.optim.Optimizer,
    rollout: RunnerRollout,
    config: RunnerPPOConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> PPODiagnostics:
    """Update on whole time sequences, minibatched only across environments."""

    policy.train()
    advantages = _normalise_advantages(rollout.advantages)
    environment_indices = np.arange(rollout.env_count)
    totals: dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "objective_aux_loss": 0.0,
        "danger_aux_loss": 0.0,
        "weighted_auxiliary_loss": 0.0,
        "approximate_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    batches = 0
    epochs_completed = 0
    early_stopped = False
    for epoch in range(config.epochs):
        rng.shuffle(environment_indices)
        epoch_kls: list[float] = []
        for start in range(0, rollout.env_count, config.minibatch_envs):
            selected = environment_indices[start : start + config.minibatch_envs]
            observations = {
                key: torch.as_tensor(value[:, selected], device=device)
                for key, value in rollout.observations.items()
            }
            _assert_action_contract(observations)
            initial_hidden = rollout.initial_hidden[:, selected].to(device)
            reset_mask = torch.as_tensor(
                rollout.episode_starts[:, selected],
                device=device,
            )
            outputs = runner_sequence_outputs(
                policy,
                observations,
                initial_hidden,
                reset_mask,
            )
            predicted_values = outputs.values
            distribution = torch.distributions.Categorical(logits=outputs.logits)
            action = torch.as_tensor(rollout.actions[:, selected], device=device)
            old_log_probability = torch.as_tensor(
                rollout.old_log_probabilities[:, selected],
                device=device,
            )
            old_value = torch.as_tensor(
                rollout.old_values[:, selected],
                device=device,
            )
            advantage = torch.as_tensor(
                advantages[:, selected],
                device=device,
            )
            returns = torch.as_tensor(
                rollout.returns[:, selected],
                device=device,
            )
            log_probability = distribution.log_prob(action)
            log_ratio = log_probability - old_log_probability
            ratio = torch.exp(log_ratio)
            policy_unclipped = ratio * advantage
            policy_clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantage
            policy_loss = -torch.minimum(policy_unclipped, policy_clipped).mean()

            clipped_value = old_value + (predicted_values - old_value).clamp(
                -config.value_clip_ratio,
                config.value_clip_ratio,
            )
            value_loss = 0.5 * torch.maximum(
                (predicted_values - returns).square(),
                (clipped_value - returns).square(),
            ).mean()
            entropy = distribution.entropy().mean()
            objective_label, danger_label = public_auxiliary_labels(observations)
            objective_aux_loss = nn.functional.mse_loss(
                outputs.objective_bearing,
                objective_label,
            )
            danger_aux_loss = nn.functional.binary_cross_entropy(
                outputs.danger,
                danger_label,
            )
            weighted_auxiliary_loss = (
                config.objective_aux_coefficient * objective_aux_loss
                + config.danger_aux_coefficient * danger_aux_loss
            )
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + weighted_auxiliary_loss
            )
            for name, value in (
                ("policy loss", policy_loss),
                ("value loss", value_loss),
                ("entropy", entropy),
                ("objective auxiliary loss", objective_aux_loss),
                ("danger auxiliary loss", danger_aux_loss),
                ("weighted auxiliary loss", weighted_auxiliary_loss),
                ("approximate KL", approximate_kl),
                ("combined loss", loss),
            ):
                _finite_tensor(name, value)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in policy.parameters():
                if parameter.grad is not None:
                    _finite_tensor("gradient", parameter.grad)
            gradient_norm = nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.max_grad_norm,
                error_if_nonfinite=True,
            )
            _finite_tensor("gradient norm", torch.as_tensor(gradient_norm))
            optimizer.step()

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "objective_aux_loss": objective_aux_loss,
                "danger_aux_loss": danger_aux_loss,
                "weighted_auxiliary_loss": weighted_auxiliary_loss,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
                "gradient_norm": torch.as_tensor(gradient_norm),
            }
            for name, value in metrics.items():
                totals[name] += float(value.detach())
            epoch_kls.append(float(approximate_kl.detach()))
            batches += 1
        epochs_completed = epoch + 1
        if (
            config.target_kl > 0.0
            and epoch_kls
            and float(np.mean(epoch_kls)) > config.target_kl
        ):
            early_stopped = True
            break

    if batches <= 0:
        raise RuntimeError("PPO update produced no minibatches")
    returns_flat = rollout.returns.reshape(-1)
    old_values_flat = rollout.old_values.reshape(-1)
    variance = float(np.var(returns_flat))
    explained_variance = (
        float(1.0 - np.var(returns_flat - old_values_flat) / variance)
        if variance > 1e-8
        else 0.0
    )
    diagnostics = PPODiagnostics(
        **{name: value / batches for name, value in totals.items()},
        explained_variance=explained_variance,
        epochs_completed=epochs_completed,
        early_stopped=early_stopped,
        samples=rollout.time_steps * rollout.env_count,
    )
    if not all(
        math.isfinite(float(value))
        for value in (
            diagnostics.policy_loss,
            diagnostics.value_loss,
            diagnostics.entropy,
            diagnostics.objective_aux_loss,
            diagnostics.danger_aux_loss,
            diagnostics.weighted_auxiliary_loss,
            diagnostics.approximate_kl,
            diagnostics.clip_fraction,
            diagnostics.gradient_norm,
            diagnostics.explained_variance,
        )
    ):
        raise FloatingPointError("non-finite PPO diagnostics; checkpoint aborted")
    return diagnostics


def acceptance_gate(
    rates: Mapping[int, float],
    previous_passes: int,
) -> int:
    """Require every tier threshold in two consecutive disjoint windows."""

    complete = all(tier in rates for tier in ALL_TIERS)
    passed = complete and all(
        float(rates[tier]) >= ACCEPTANCE_THRESHOLDS[tier]
        for tier in ALL_TIERS
    )
    return previous_passes + 1 if passed else 0


def curriculum_gate(
    *,
    current_tier: int,
    rates: Mapping[int, float],
    previous_passes: int,
) -> tuple[int, int, bool]:
    """Advance a tier only after two consecutive held-out passes."""

    current_tier = int(current_tier)
    if current_tier not in ALL_TIERS:
        raise ValueError("current_tier must lie in 1..6")
    success = float(rates.get(current_tier, 0.0))
    passes = (
        previous_passes + 1
        if success >= PROMOTION_THRESHOLDS[current_tier]
        else 0
    )
    if passes >= 2 and current_tier < 6:
        return current_tier + 1, 0, True
    return current_tier, passes, False


def validation_selection_key(
    report: Mapping[str, Any],
    *,
    required_tiers: Sequence[int] = ALL_TIERS,
) -> tuple[float, float, float, float]:
    tiers = {
        int(tier): values
        for tier, values in dict(report.get("tiers", {})).items()
    }
    selected_tiers = parse_tiers(required_tiers)
    if any(tier not in tiers for tier in selected_tiers):
        return (-1.0, -1.0, -math.inf, -math.inf)
    successes = {
        tier: float(tiers[tier]["success_rate"])
        for tier in selected_tiers
    }
    mean_damage = float(
        np.mean(
            [
                float(tiers[tier]["mean_damage"])
                for tier in selected_tiers
            ]
        )
    )
    mean_duration = float(
        np.mean(
            [
                float(tiers[tier]["mean_duration_seconds"])
                for tier in selected_tiers
            ]
        )
    )
    return (
        min(successes.values()),
        successes.get(6, successes[max(selected_tiers)]),
        -mean_damage,
        -mean_duration,
    )


def selection_validation_tiers(config: RunnerPPOConfig) -> tuple[int, ...]:
    """Keep checkpoint ranking independent from the training curriculum."""

    return config.tiers


@torch.no_grad()
def validate_runner(
    policy: RunnerPolicyV2,
    *,
    episodes_per_tier: int,
    validation_cursor: int,
    device: torch.device,
    tiers: Sequence[int] = ALL_TIERS,
    directive: ContractDirective = ContractDirective.STANDARD,
    batch_size: int = 16,
    security_pool: Any | None = None,
    ghost_training_stage: int = 0,
) -> dict[str, Any]:
    """Evaluate a deterministic policy on one reserved, disjoint seed window."""

    require_validation_window(validation_cursor, episodes_per_tier)
    if batch_size <= 0:
        raise ValueError("validation batch_size must be positive")
    selected_tiers = parse_tiers(tiers)
    was_training = policy.training
    policy.eval()
    summaries: dict[str, dict[str, float | int]] = {}
    for tier in selected_tiers:
        successes = 0
        damages: list[float] = []
        durations: list[float] = []
        for batch_start in range(0, episodes_per_tier, batch_size):
            batch_end = min(episodes_per_tier, batch_start + batch_size)
            environments: list[GhostlineEnvV2] = []
            observations: list[dict[str, np.ndarray]] = []
            try:
                for episode in range(batch_start, batch_end):
                    seed = validation_seed(tier, validation_cursor + episode)
                    if security_pool is None:
                        env = GhostlineEnvV2(
                            seed=seed,
                            tier=tier,
                            directive=directive,
                        )
                    else:
                        from ghostline.security_opponents import (
                            FrozenSecurityRunnerEnvV2,
                        )

                        env = FrozenSecurityRunnerEnvV2(
                            security_pool=security_pool,
                            seed=seed,
                            tier=tier,
                            directive=directive,
                        )
                    observation, _ = env.reset(
                        seed=seed,
                        options={"tier": tier, "directive": directive},
                    )
                    observation = apply_training_ghost_security_stage(
                        env,
                        stage=ghost_training_stage,
                        tier=tier,
                        directive=directive,
                        observation=observation,
                    )
                    environments.append(env)
                    observations.append(observation)
                hidden = torch.zeros(
                    1,
                    len(environments),
                    policy.recurrent_size,
                    device=device,
                )
                while environments:
                    stacked = _stack_observations(observations)
                    tensors = _tensor_observation(stacked, device)
                    _assert_action_contract(tensors)
                    logits, _, next_hidden = policy(tensors, hidden)
                    actions = torch.argmax(logits, dim=-1).cpu().numpy()
                    survivors: list[int] = []
                    next_observations: list[dict[str, np.ndarray]] = []
                    next_environments: list[GhostlineEnvV2] = []
                    for index, (env, action) in enumerate(
                        zip(environments, actions, strict=True)
                    ):
                        observation, _, terminated, truncated, info = env.step(
                            int(action)
                        )
                        if terminated or truncated:
                            successes += int(bool(info.get("is_success", False)))
                            damages.append(
                                float(info.get("damage", env.sim.damage_taken))
                            )
                            durations.append(
                                float(
                                    info.get(
                                        "duration_seconds",
                                        env.sim.elapsed_seconds,
                                    )
                                )
                            )
                            env.close()
                        else:
                            survivors.append(index)
                            next_environments.append(env)
                            next_observations.append(observation)
                    environments = next_environments
                    observations = next_observations
                    if survivors:
                        hidden = next_hidden[:, survivors, :]
            finally:
                for env in environments:
                    env.close()
        summaries[str(tier)] = {
            "episodes": episodes_per_tier,
            "successes": successes,
            "success_rate": successes / episodes_per_tier,
            "mean_damage": float(np.mean(damages)),
            "mean_duration_seconds": float(np.mean(durations)),
        }
    policy.train(was_training)
    return {
        "contract": OBSERVATION_CONTRACT_V2,
        "environment_fingerprint": multi_agent_environment_fingerprint(),
        "model_fingerprint": runner_model_fingerprint(),
        "validation_cursor": validation_cursor,
        "episodes_per_tier": episodes_per_tier,
        "directive": directive.name.lower(),
        "security_opponent_pool": (
            getattr(security_pool, "pool_id", None)
        ),
        "training_only_ghost_security_stage": int(ghost_training_stage),
        "tiers": summaries,
    }


@torch.no_grad()
def validate_runner_suite(
    policy: RunnerPolicyV2,
    *,
    episodes_per_tier: int,
    validation_cursor: int,
    device: torch.device,
    tiers: Sequence[int] = ALL_TIERS,
    directives: Sequence[ContractDirective | str | int] = ALL_DIRECTIVES,
    batch_size: int = 16,
    security_opponent_paths: Sequence[str | Path] = (),
    security_pool_salt: int = 0,
    ghost_training_stage: int = 0,
) -> dict[str, Any]:
    """Gate a universal policy by its worst result across trained directives."""

    selected_directives = parse_directives(directives)
    security_pool: Any | None = None
    if security_opponent_paths:
        from ghostline.security_opponents import FrozenSecurityOpponentPool

        security_pool = FrozenSecurityOpponentPool(
            tuple(Path(path) for path in security_opponent_paths),
            selection_salt=security_pool_salt,
        )
    try:
        reports = {
            directive.name.lower(): validate_runner(
                policy,
                episodes_per_tier=episodes_per_tier,
                validation_cursor=validation_cursor,
                device=device,
                tiers=tiers,
                directive=directive,
                batch_size=batch_size,
                security_pool=security_pool,
                ghost_training_stage=ghost_training_stage,
            )
            for directive in selected_directives
        }
    finally:
        if security_pool is not None:
            security_pool.close()
    selected_tiers = parse_tiers(tiers)
    summaries: dict[str, dict[str, float | int | str]] = {}
    for tier in selected_tiers:
        records = [
            (
                name,
                report["tiers"][str(tier)],
            )
            for name, report in reports.items()
        ]
        worst_name, worst = min(
            records,
            key=lambda item: (
                float(item[1]["success_rate"]),
                item[0],
            ),
        )
        total_successes = sum(int(record["successes"]) for _, record in records)
        total_episodes = episodes_per_tier * len(records)
        summaries[str(tier)] = {
            "episodes": total_episodes,
            "episodes_per_directive": episodes_per_tier,
            "successes": total_successes,
            # Curriculum and checkpoint selection intentionally use the
            # weakest directive, not an average that can hide a collapse.
            "success_rate": float(worst["success_rate"]),
            "mean_success_rate": total_successes / total_episodes,
            "worst_directive": worst_name,
            "mean_damage": float(
                np.mean(
                    [float(record["mean_damage"]) for _, record in records]
                )
            ),
            "mean_duration_seconds": float(
                np.mean(
                    [
                        float(record["mean_duration_seconds"])
                        for _, record in records
                    ]
                )
            ),
        }
    return {
        "contract": OBSERVATION_CONTRACT_V2,
        "environment_fingerprint": multi_agent_environment_fingerprint(),
        "model_fingerprint": runner_model_fingerprint(),
        "validation_cursor": validation_cursor,
        "episodes_per_tier_per_directive": episodes_per_tier,
        "evaluated_directives": tuple(reports),
        "directives": reports,
        "security_opponent_pool": (
            reports[next(iter(reports))].get("security_opponent_pool")
            if reports
            else None
        ),
        "training_only_ghost_security_stage": int(ghost_training_stage),
        "tiers": summaries,
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Durably replace a checkpoint without exposing a partial file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _policy_payload(
    policy: RunnerPolicyV2,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": policy.state_dict(),
        "recurrent_size": policy.recurrent_size,
        "observation_contract": OBSERVATION_CONTRACT_V2,
        "action_count": RUNNER_ACTION_COUNT_V2,
        "environment_fingerprint": multi_agent_environment_fingerprint(),
        "model_fingerprint": runner_model_fingerprint(),
        "metadata": dict(metadata),
    }


def _checkpoint_contract(config: RunnerPPOConfig) -> dict[str, Any]:
    return {
        "trainer_contract": TRAINER_CONTRACT_V2,
        "config": config.contract(),
        "observation_contract": OBSERVATION_CONTRACT_V2,
        "action_count": RUNNER_ACTION_COUNT_V2,
        "environment_fingerprint": multi_agent_environment_fingerprint(),
        "model_fingerprint": runner_model_fingerprint(),
        "trainer_fingerprint": runner_trainer_fingerprint(),
        "runtime": runtime_fingerprint(),
    }


def _capture_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "numpy_global": np.random.get_state(),
        "numpy_generator": deepcopy(rng.bit_generator.state),
    }


def _restore_rng_state(
    state: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    required = ("torch", "numpy_global", "numpy_generator")
    missing = [name for name in required if name not in state]
    if missing:
        raise RuntimeError(f"runner checkpoint omitted RNG state: {missing}")
    torch.set_rng_state(torch.as_tensor(state["torch"], device="cpu").to(torch.uint8))
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint owns CUDA RNG states but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, device="cpu").to(torch.uint8) for item in cuda_states]
        )
    np.random.set_state(state["numpy_global"])
    rng.bit_generator.state = deepcopy(state["numpy_generator"])


def save_training_checkpoint(
    path: Path,
    *,
    policy: RunnerPolicyV2,
    optimizer: torch.optim.Optimizer,
    config: RunnerPPOConfig,
    rng: np.random.Generator,
    vector_env: gym.vector.VectorEnv,
    observation: Mapping[str, np.ndarray],
    hidden: torch.Tensor,
    episode_starts: np.ndarray,
    updates: int,
    decisions: int,
    validation_cursor: int,
    acceptance_passes: int,
    curriculum_tier: int,
    promotion_passes: int,
    best_selection_key: tuple[float, float, float, float],
    next_validation_update: int,
    validation_history: Sequence[Mapping[str, Any]],
    initialization: Mapping[str, Any],
) -> None:
    """Save a strict resume point, including every in-progress environment."""

    states = vector_env.call("checkpoint_state")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "contract": _checkpoint_contract(config),
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "updates": int(updates),
        "decisions": int(decisions),
        "validation_cursor": int(validation_cursor),
        "acceptance_passes": int(acceptance_passes),
        "curriculum_tier": int(curriculum_tier),
        "promotion_passes": int(promotion_passes),
        "best_selection_key": tuple(float(value) for value in best_selection_key),
        "next_validation_update": int(next_validation_update),
        "validation_history": list(validation_history),
        "initialization": dict(initialization),
        "hidden": hidden.detach().cpu(),
        "episode_starts": np.asarray(episode_starts, dtype=bool),
        "vector_states": list(states),
        "observation_digest": observation_digest(observation),
        "rng": _capture_rng_state(rng),
    }
    _atomic_torch_save(payload, Path(path))


def _require_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    config: RunnerPPOConfig,
) -> None:
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise RuntimeError("runner checkpoint version changed")
    expected = _checkpoint_contract(config)
    actual = checkpoint.get("contract")
    if not isinstance(actual, Mapping):
        raise RuntimeError("runner checkpoint predates the strict resume contract")
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"runner resume contract changed: {mismatches}")


def load_training_checkpoint(
    path: Path,
    *,
    policy: RunnerPolicyV2,
    optimizer: torch.optim.Optimizer,
    config: RunnerPPOConfig,
    rng: np.random.Generator,
    vector_env: gym.vector.VectorEnv,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint and deterministically reconstruct all live workers."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("runner checkpoint payload is not a mapping")
    _require_checkpoint_contract(checkpoint, config)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    states = checkpoint.get("vector_states")
    if not isinstance(states, Sequence) or len(states) != config.envs:
        raise RuntimeError("runner checkpoint vector state count changed")
    checkpoint_curriculum_tier = int(checkpoint["curriculum_tier"])
    if checkpoint_curriculum_tier not in ALL_TIERS:
        raise RuntimeError("runner checkpoint curriculum tier is invalid")
    if any(
        int(state.get("curriculum_tier", -1)) != checkpoint_curriculum_tier
        for state in states
    ):
        raise RuntimeError("runner checkpoint workers disagree on curriculum tier")
    restored = vector_env.call("restore_state", states)
    observation = _stack_observations(restored)
    if observation_digest(observation) != checkpoint.get("observation_digest"):
        raise RuntimeError("restored vector observation differs from checkpoint")
    hidden = torch.as_tensor(checkpoint["hidden"], device=device)
    if hidden.shape != (1, config.envs, config.recurrent_size):
        raise RuntimeError("runner checkpoint hidden-state shape changed")
    episode_starts = np.asarray(checkpoint["episode_starts"], dtype=bool)
    if episode_starts.shape != (config.envs,):
        raise RuntimeError("runner checkpoint episode-start shape changed")
    _restore_rng_state(checkpoint["rng"], rng)
    initialization = checkpoint.get("initialization")
    if not isinstance(initialization, Mapping) or "method" not in initialization:
        raise RuntimeError("runner checkpoint omitted initialization provenance")
    return {
        "observation": observation,
        "hidden": hidden,
        "episode_starts": episode_starts,
        "updates": int(checkpoint["updates"]),
        "decisions": int(checkpoint["decisions"]),
        "validation_cursor": int(checkpoint["validation_cursor"]),
        "acceptance_passes": int(checkpoint["acceptance_passes"]),
        "curriculum_tier": checkpoint_curriculum_tier,
        "promotion_passes": int(checkpoint["promotion_passes"]),
        "best_selection_key": tuple(checkpoint["best_selection_key"]),
        "next_validation_update": int(checkpoint["next_validation_update"]),
        "validation_history": list(checkpoint["validation_history"]),
        "initialization": dict(initialization),
    }


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def initialize_fresh_policy(
    args: argparse.Namespace,
    config: RunnerPPOConfig,
    device: torch.device,
) -> tuple[RunnerPolicyV2, dict[str, Any]]:
    """Construct a fresh v2 policy and bind its initialization provenance."""

    resume = bool(getattr(args, "resume", False))
    init_checkpoint = getattr(args, "init_checkpoint", None)
    published_v1_init = getattr(args, "published_v1_init", None)
    if resume:
        raise ValueError("initialize_fresh_policy cannot be used for a resume")
    if init_checkpoint is not None and published_v1_init is not None:
        raise ValueError(
            "--init-checkpoint and --published-v1-init are mutually exclusive"
        )
    if published_v1_init is not None:
        source = Path(published_v1_init)
        policy = RunnerPolicyV2(recurrent_size=config.recurrent_size).to(device)
        report = initialize_runner_v2_from_published_v1(
            policy,
            source,
            value_scale=config.reward_scale,
        )
        return policy.train(), {
            **report,
            "source": str(source.resolve()),
            "source_sha256": _sha256(source),
        }
    if init_checkpoint is not None:
        source = Path(init_checkpoint)
        policy = load_runner_v2(source, device=device)
        if policy.recurrent_size != config.recurrent_size:
            raise ValueError("initial runner checkpoint recurrent size differs")
        return policy.train(), {
            "method": "runner-v2-checkpoint",
            "source": str(source.resolve()),
            "source_sha256": _sha256(source),
            "source_observation_contract": OBSERVATION_CONTRACT_V2,
            "source_recurrent_size": policy.recurrent_size,
        }
    return RunnerPolicyV2(recurrent_size=config.recurrent_size).to(device), {
        "method": "orthogonal-scratch",
        "seed": config.seed,
        "target_observation_contract": OBSERVATION_CONTRACT_V2,
        "target_action_count": RUNNER_ACTION_COUNT_V2,
        "target_recurrent_size": config.recurrent_size,
    }


def train(
    args: argparse.Namespace,
) -> Path:
    """Run PPO until its explicit update/decision/time budget or acceptance."""

    tiers = parse_tiers(args.tiers)
    directives = parse_directives(args.directives)
    raw_security_opponents = tuple(
        Path(path).expanduser().resolve()
        for path in (getattr(args, "security_opponent", None) or ())
    )
    if len(set(raw_security_opponents)) != len(raw_security_opponents):
        raise ValueError("duplicate --security-opponent checkpoints are not allowed")
    for path in raw_security_opponents:
        if not path.is_file():
            raise FileNotFoundError(
                f"security opponent checkpoint is missing: {path}"
            )
    config = RunnerPPOConfig(
        seed=args.seed,
        envs=args.envs,
        rollout=args.rollout,
        epochs=args.epochs,
        minibatch_envs=args.minibatch_envs,
        recurrent_size=args.recurrent_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        reward_scale=float(getattr(args, "reward_scale", 0.05)),
        clip_ratio=args.clip_ratio,
        value_clip_ratio=args.value_clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        objective_aux_coefficient=float(
            getattr(args, "objective_aux_coefficient", 0.15)
        ),
        danger_aux_coefficient=float(
            getattr(args, "danger_aux_coefficient", 0.10)
        ),
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        training_seed_start=args.training_seed_start,
        tiers=tiers,
        directives=tuple(int(item) for item in directives),
        ghost_directive_fraction=float(
            getattr(args, "ghost_directive_fraction", 0.25)
        ),
        ghost_training_stage=int(
            getattr(args, "ghost_training_stage", 0)
        ),
        adaptive_curriculum=bool(getattr(args, "adaptive_curriculum", True)),
        initial_curriculum_tier=int(getattr(args, "initial_curriculum_tier", 1)),
        async_envs=args.async_envs,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        validation_batch_size=int(getattr(args, "validation_batch_size", 16)),
        initial_validation_cursor=args.initial_validation_cursor,
        checkpoint_interval=args.checkpoint_interval,
        security_opponent_paths=tuple(
            str(path) for path in raw_security_opponents
        ),
        security_opponent_sha256=tuple(
            _sha256(path) for path in raw_security_opponents
        ),
        security_pool_salt=int(
            getattr(args, "security_pool_salt", 0)
        ),
    )
    config.validate()
    output = Path(args.output)
    latest_path = output / "latest.pt"
    champion_path = output / "best.pt"
    last_policy_path = output / "last-policy.pt"
    metrics_path = output / "metrics.jsonl"
    manifest_path = output / "experiment-manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    existing_outputs = tuple(
        path
        for path in (latest_path, champion_path, last_policy_path)
        if path.exists()
    )
    if not args.resume and existing_outputs:
        raise RuntimeError(
            f"{output} already contains runner checkpoints; use --resume or a new output"
        )
    if args.resume and not latest_path.exists():
        raise RuntimeError(f"cannot resume because {latest_path} does not exist")
    if args.resume and (
        args.init_checkpoint is not None
        or getattr(args, "published_v1_init", None) is not None
    ):
        raise ValueError(
            "--resume cannot be combined with an initialization checkpoint"
        )
    if (
        not args.resume
        and args.init_checkpoint is None
        and getattr(args, "published_v1_init", None) is None
        and not bool(getattr(args, "allow_scratch", False))
    ):
        raise ValueError(
            "fresh v2 PPO requires --published-v1-init or --init-checkpoint; "
            "use --allow-scratch only for an intentional ablation"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    if bool(getattr(args, "dry_run", False)):
        resume_checkpoint: Path | None = None
        if args.resume:
            checkpoint = torch.load(
                latest_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(checkpoint, Mapping):
                raise RuntimeError(
                    "runner checkpoint payload is not a mapping"
                )
            _require_checkpoint_contract(checkpoint, config)
            initialization = dict(checkpoint["initialization"])
            resume_checkpoint = latest_path
        else:
            dry_policy, initialization = initialize_fresh_policy(
                args,
                config,
                torch.device("cpu"),
            )
            del dry_policy
        _atomic_json_save(
            build_experiment_manifest(
                args=args,
                config=config,
                initialization=initialization,
                device=device,
                status="preflight-passed",
                resume_checkpoint=resume_checkpoint,
            ),
            manifest_path,
        )
        return manifest_path

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed ^ 0xA0761D6478BD642F)

    if args.resume:
        policy = RunnerPolicyV2(recurrent_size=config.recurrent_size).to(device)
        initialization: dict[str, Any] = {}
    else:
        policy, initialization = initialize_fresh_policy(
            args,
            config,
            device,
        )
        _atomic_json_save(
            build_experiment_manifest(
                args=args,
                config=config,
                initialization=initialization,
                device=device,
                status="training-started",
            ),
            manifest_path,
        )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
        weight_decay=1e-4,
    )
    vector_env = make_runner_vector_env(config)

    updates = 0
    decisions = 0
    validation_cursor = config.initial_validation_cursor
    acceptance_passes = 0
    curriculum_tier = config.initial_curriculum_tier
    promotion_passes = 0
    best_selection_key = (-1.0, -1.0, -math.inf, -math.inf)
    next_validation_update = (
        config.validation_interval
        if config.validation_interval
        else 2**63 - 1
    )
    validation_history: list[Mapping[str, Any]] = []
    if args.resume:
        state = load_training_checkpoint(
            latest_path,
            policy=policy,
            optimizer=optimizer,
            config=config,
            rng=rng,
            vector_env=vector_env,
            device=device,
        )
        observation = state["observation"]
        hidden = state["hidden"]
        episode_starts = state["episode_starts"]
        updates = state["updates"]
        decisions = state["decisions"]
        validation_cursor = state["validation_cursor"]
        acceptance_passes = state["acceptance_passes"]
        curriculum_tier = state["curriculum_tier"]
        promotion_passes = state["promotion_passes"]
        best_selection_key = state["best_selection_key"]
        next_validation_update = state["next_validation_update"]
        validation_history = state["validation_history"]
        initialization = state["initialization"]
        if not manifest_path.exists():
            _atomic_json_save(
                build_experiment_manifest(
                    args=args,
                    config=config,
                    initialization=initialization,
                    device=device,
                    status="resume-recovered-manifest",
                    resume_checkpoint=latest_path,
                ),
                manifest_path,
            )
    else:
        observation, _ = vector_env.reset()
        observation = {
            key: np.asarray(observation[key]) for key in OBSERVATION_KEYS_V2
        }
        hidden = torch.zeros(
            1,
            config.envs,
            config.recurrent_size,
            device=device,
        )
        episode_starts = np.ones(config.envs, dtype=bool)

    start_time = time.monotonic()
    start_decisions = decisions
    recent_successes: list[float] = []

    def should_continue() -> bool:
        if args.max_updates > 0 and updates >= args.max_updates:
            return False
        if args.max_decisions > 0 and decisions >= args.max_decisions:
            return False
        if args.seconds > 0.0 and time.monotonic() - start_time >= args.seconds:
            return False
        if args.stop_on_acceptance and acceptance_passes >= 2:
            return False
        return True

    try:
        while should_continue():
            rollout = collect_rollout(
                policy=policy,
                vector_env=vector_env,
                observation=observation,
                hidden=hidden,
                episode_starts=episode_starts,
                rollout_steps=config.rollout,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                reward_scale=config.reward_scale,
                device=device,
            )
            diagnostics = ppo_update(
                policy=policy,
                optimizer=optimizer,
                rollout=rollout,
                config=config,
                rng=rng,
                device=device,
            )
            observation = rollout.next_observation
            hidden = rollout.next_hidden
            episode_starts = rollout.next_episode_starts
            updates += 1
            decisions += rollout.time_steps * rollout.env_count
            recent_successes.extend(rollout.completed_successes)
            recent_successes = recent_successes[-200:]

            validation_report: Mapping[str, Any] | None = None
            if config.validation_interval and updates >= next_validation_update:
                require_validation_window(validation_cursor, config.validation_episodes)
                validation_report = validate_runner_suite(
                    policy,
                    episodes_per_tier=config.validation_episodes,
                    validation_cursor=validation_cursor,
                    device=device,
                    # Curriculum limits the training distribution, never the
                    # held-out selection distribution.  A league generation
                    # must be rankable from its first validation even when
                    # promotion has not yet reached tier 6.
                    tiers=selection_validation_tiers(config),
                    directives=tuple(
                        ContractDirective(value)
                        for value in config.directives
                    ),
                    batch_size=config.validation_batch_size,
                    security_opponent_paths=config.security_opponent_paths,
                    security_pool_salt=config.security_pool_salt,
                    ghost_training_stage=config.ghost_training_stage,
                )
                validation_history.append(validation_report)
                rates = {
                    int(tier): float(summary["success_rate"])
                    for tier, summary in validation_report["tiers"].items()
                }
                if config.adaptive_curriculum:
                    previous_tier = curriculum_tier
                    curriculum_tier, promotion_passes, promoted = curriculum_gate(
                        current_tier=curriculum_tier,
                        rates=rates,
                        previous_passes=promotion_passes,
                    )
                    if promoted:
                        vector_env.call("set_curriculum_tier", curriculum_tier)
                        acceptance_passes = 0
                    elif previous_tier == 6:
                        acceptance_passes = acceptance_gate(
                            rates,
                            acceptance_passes,
                        )
                    else:
                        acceptance_passes = 0
                else:
                    acceptance_passes = acceptance_gate(rates, acceptance_passes)
                selection_key = validation_selection_key(
                    validation_report,
                    required_tiers=config.tiers,
                )
                if selection_key > best_selection_key:
                    best_selection_key = selection_key
                    _atomic_torch_save(
                        _policy_payload(
                            policy,
                            metadata={
                                "updates": updates,
                                "decisions": decisions,
                                "selection_key": selection_key,
                                "validation": validation_report,
                                "purpose": "runner-v2-validation-champion",
                                "initialization": initialization,
                                "training_config": config.contract(),
                                "trainer_contract": TRAINER_CONTRACT_V2,
                                "trainer_fingerprint": runner_trainer_fingerprint(),
                            },
                        ),
                        champion_path,
                    )
                validation_cursor += config.validation_episodes
                if acceptance_passes < 2:
                    require_validation_window(
                        validation_cursor,
                        config.validation_episodes,
                    )
                next_validation_update = updates + config.validation_interval

            elapsed = max(1e-9, time.monotonic() - start_time)
            record: dict[str, Any] = {
                "updates": updates,
                "decisions": decisions,
                "session_decisions_per_second": (
                    decisions - start_decisions
                ) / elapsed,
                "recent_success_rate": (
                    float(np.mean(recent_successes))
                    if recent_successes
                    else 0.0
                ),
                "episodes_finished": len(recent_successes),
                **diagnostics.as_dict(),
            }
            if validation_report is not None:
                record["validation_cursor"] = validation_report["validation_cursor"]
                record["validation"] = validation_report["tiers"]
                record["acceptance_passes"] = acceptance_passes
                record["curriculum_tier"] = curriculum_tier
                record["promotion_passes"] = promotion_passes
            _append_jsonl(metrics_path, record)

            if (
                updates % config.checkpoint_interval == 0
                or not should_continue()
            ):
                save_training_checkpoint(
                    latest_path,
                    policy=policy,
                    optimizer=optimizer,
                    config=config,
                    rng=rng,
                    vector_env=vector_env,
                    observation=observation,
                    hidden=hidden,
                    episode_starts=episode_starts,
                    updates=updates,
                    decisions=decisions,
                    validation_cursor=validation_cursor,
                    acceptance_passes=acceptance_passes,
                    curriculum_tier=curriculum_tier,
                    promotion_passes=promotion_passes,
                    best_selection_key=best_selection_key,
                    next_validation_update=next_validation_update,
                    validation_history=validation_history,
                    initialization=initialization,
                )
    finally:
        vector_env.close()

    _atomic_torch_save(
        _policy_payload(
            policy,
            metadata={
                "updates": updates,
                "decisions": decisions,
                "purpose": "runner-v2-last-policy",
                "initialization": initialization,
                "training_config": config.contract(),
                "trainer_contract": TRAINER_CONTRACT_V2,
                "trainer_fingerprint": runner_trainer_fingerprint(),
            },
        ),
        last_policy_path,
    )
    return champion_path if champion_path.exists() else last_policy_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the GhostlineEnv-v2 recurrent runner with clipped PPO",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/runner-v2/ppo"))
    parser.add_argument("--resume", action="store_true")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", type=Path)
    initialization.add_argument(
        "--published-v1-init",
        type=Path,
        metavar="CHECKPOINT",
        help="transplant shared weights from the immutable published-v1 policy",
    )
    parser.add_argument(
        "--allow-scratch",
        action="store_true",
        help="allow an explicit scratch-PPO ablation instead of a trained initialization",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-envs", type=int, default=2)
    parser.add_argument("--recurrent-size", type=int, choices=(256, 384, 512), default=384)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--objective-aux-coefficient", type=float, default=0.15)
    parser.add_argument("--danger-aux-coefficient", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--training-seed-start", type=int, default=TRAINING_SEED_START)
    parser.add_argument("--tiers", default="1,2,3,4,5,6")
    parser.add_argument("--directives", default="standard,ghost,speed,greed")
    parser.add_argument(
        "--ghost-directive-fraction",
        type=float,
        default=0.25,
        help="training share reserved for ghost contracts; validation stays balanced",
    )
    parser.add_argument(
        "--ghost-training-stage",
        type=int,
        choices=tuple(GHOST_TRAINING_STAGE_COUNTS),
        default=0,
        help=(
            "training-only Ghost roster: 1=one guard, 2=one guard+camera, "
            "3=two guards+camera; 0 keeps the full release environment"
        ),
    )
    parser.add_argument(
        "--no-curriculum",
        dest="adaptive_curriculum",
        action="store_false",
        help="sample the requested tiers uniformly instead of promoting on validation",
    )
    parser.set_defaults(adaptive_curriculum=True)
    parser.add_argument(
        "--initial-curriculum-tier",
        type=int,
        choices=ALL_TIERS,
        default=1,
    )
    parser.add_argument(
        "--sync-envs",
        dest="async_envs",
        action="store_false",
        help="use in-process vector workers (primarily for tests/debugging)",
    )
    parser.set_defaults(async_envs=True)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=25)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--initial-validation-cursor", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--security-opponent",
        type=Path,
        action="append",
        default=[],
        metavar="CHECKPOINT",
        help="repeatable frozen v2 security checkpoint for deterministic opponent-pool training",
    )
    parser.add_argument("--security-pool-salt", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the campaign and write experiment-manifest.json without starting workers",
    )
    parser.add_argument(
        "--continue-after-acceptance",
        dest="stop_on_acceptance",
        action="store_false",
        help="continue after two complete held-out acceptance passes",
    )
    parser.set_defaults(stop_on_acceptance=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(args: Sequence[str] | None = None) -> Path:
    return train(build_parser().parse_args(args))


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``.
    main()
