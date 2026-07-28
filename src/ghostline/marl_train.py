"""Parameter-shared recurrent MAPPO for Ghostline's adaptive security team."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from ghostline.config_v2 import MAX_SECURITY_TARGETS
from ghostline.security_env import (
    SECURITY_REWARD_GAMMA,
    GhostlineSecurityParallelEnv,
)
from ghostline.runner_opponents import (
    FrozenPublishedV1RunnerOpponent,
    FrozenRunnerV2Opponent,
    load_runner_opponent_policy,
    make_frozen_runner_opponent,
    runner_opponent_kind,
)
from ghostline.security_baselines import tactical_security_action
from ghostline.security_model import (
    SECURITY_MASK_KEYS,
    SECURITY_OBSERVATION_CONTRACT,
    SECURITY_ACTION_FACTORS,
    SECURITY_ACTION_SIZES,
    SharedSecurityActorCritic,
    factorized_log_prob,
    load_security_policy,
    save_security_policy,
    security_environment_fingerprint,
    select_factorized_actions,
)


SECURITY_TRAIN_SEED_START = 10_000_000
SECURITY_VALIDATION_SEED_START = 11_000_000
SECURITY_FINAL_TEST_SEED_START = 14_000_000
DEFAULT_SECURITY_FINAL_SLICE_MANIFEST = Path(
    "benchmarks/security/v2-final-test-slices.json"
)
SECURITY_EXPERIMENT_MANIFEST_CONTRACT = (
    "ghostline-security-v2-experiment-manifest-v1"
)
MAX_OPERATIVES = 5
ACTOR_OBS_KEYS = (
    "ego",
    "local_grid",
    "runner",
    "teammates",
    "teammate_mask",
    "targets",
    "target_mask",
    "intent_target_mask",
    "radio",
    "radio_mask",
    "intent_mask",
    "message_mask",
    "ability_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _source_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain=v1", "--untracked-files=normal")
        return {
            "available": True,
            "repository": str(root),
            "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(
                status.encode("utf-8")
            ).hexdigest(),
        }
    except (FileNotFoundError, subprocess.SubprocessError):
        return {
            "available": False,
            "repository": str(root),
            "commit": None,
            "branch": None,
            "dirty": None,
        }


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
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


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _selection_key(report: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    tiers = report["tiers"]
    summaries = list(tiers.values())
    tier_six = tiers.get("6", summaries[-1])
    return (
        float(report["worst_tier_security_stop_rate"]),
        float(tier_six["security_stop_rate"]),
        sum(float(item["security_stop_rate"]) for item in summaries) / len(summaries),
        sum(float(item["mean_damage"]) for item in summaries) / len(summaries),
        sum(float(item["mean_detections"]) for item in summaries) / len(summaries),
        sum(float(item["mean_duration_seconds"]) for item in summaries) / len(summaries),
    )


def parse_security_tiers(value: str | Iterable[int]) -> tuple[int, ...]:
    tiers = tuple(int(item) for item in value.split(",")) if isinstance(value, str) else tuple(int(item) for item in value)
    if not tiers or any(tier not in range(3, 7) for tier in tiers):
        raise ValueError("security tiers must be a comma-separated subset of 3,4,5,6")
    if len(set(tiers)) != len(tiers):
        raise ValueError("security tiers must not contain duplicates")
    return tiers


def _adaptive_tier_probabilities(report: dict[str, Any], tiers: tuple[int, ...]) -> np.ndarray:
    """Allocate 70% replay to the current weakest held-out tier set."""

    rates = np.asarray([float(report["tiers"][str(tier)]["security_stop_rate"]) for tier in tiers])
    weakest = np.isclose(rates, rates.min(), atol=1e-9)
    probabilities = np.full(len(tiers), 0.30 / len(tiers), dtype=np.float64)
    probabilities[weakest] += 0.70 / max(1, int(weakest.sum()))
    return probabilities / probabilities.sum()


def _padded_observations(
    envs: list[GhostlineSecurityParallelEnv],
    observations: list[dict[str, dict[str, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sample_space = envs[0]._observation_space  # Stable internal batch contract.
    result = {
        key: np.zeros((len(envs), MAX_OPERATIVES, *sample_space[key].shape), dtype=sample_space[key].dtype)
        for key in ACTOR_OBS_KEYS
    }
    active = np.zeros((len(envs), MAX_OPERATIVES), dtype=np.float32)
    for env_index, (env, records) in enumerate(zip(envs, observations, strict=True)):
        for agent, observation in records.items():
            slot = env.agent_name_mapping[agent]
            active[env_index, slot] = 1.0
            for key in ACTOR_OBS_KEYS:
                result[key][env_index, slot] = observation[key]
    # Categorical distributions for padded agents still need one finite logit.
    for key in SECURITY_MASK_KEYS:
        empty = result[key].sum(axis=-1) == 0
        result[key][empty, 0] = 1
    return result, active


def _actor_tensors(observation: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device).flatten(0, 1)
        for key, value in observation.items()
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _normalize_active_advantages(
    advantages: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Normalize strictly over active operative transitions.

    Procedural tiers use three or five operatives. Padded slots can contain
    arbitrary critic values, so including them in rollout statistics changes
    the gradient merely because a smaller team was sampled.
    """

    values = np.asarray(advantages, dtype=np.float32)
    mask = np.asarray(active, dtype=np.float32) > 0.0
    if values.shape != mask.shape:
        raise ValueError("advantages and active mask must have identical shapes")
    result = np.zeros_like(values, dtype=np.float32)
    selected = values[mask]
    if selected.size == 0:
        return result
    mean = float(selected.mean(dtype=np.float64))
    standard_deviation = float(selected.std(dtype=np.float64))
    result[mask] = (selected - mean) / max(1e-6, standard_deviation)
    return result


def _masked_value_loss(
    predicted: torch.Tensor,
    old: torch.Tensor,
    returns: torch.Tensor,
    active: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    """Clipped PPO value loss with no padded-agent contribution."""

    clipped = old + (predicted - old).clamp(-clip_ratio, clip_ratio)
    losses = 0.5 * torch.maximum(
        (predicted - returns).square(),
        (clipped - returns).square(),
    )
    return _masked_mean(losses, active.float())


def _sample_actions(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    intent_target_mask: torch.Tensor,
) -> torch.Tensor:
    return select_factorized_actions(
        logits,
        intent_target_mask,
        deterministic=False,
    )



class RunningReturnScale:
    """Legacy diagnostic retained for old reports; training no longer uses it.

    The former trainer changed this scale between rollout and optimization
    without PopArt output preservation and did not checkpoint it. The bounded
    v2 reward contract instead trains the critic directly in reward units.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.mean_square = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        count = flat.size
        total = self.count + count
        self.mean += (flat.mean() - self.mean) * count / total
        self.mean_square += (np.square(flat).mean() - self.mean_square) * count / total
        self.count = total

    @property
    def sigma(self) -> float:
        return float(max(1e-6, np.sqrt(max(1e-12, self.mean_square))))


@torch.no_grad()
def _batched_security_actions(
    policy: SharedSecurityActorCritic,
    observations: dict[str, dict[str, np.ndarray]],
    hidden: torch.Tensor | None,
    *,
    deterministic: bool,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], torch.Tensor]:
    """Evaluate every active operative in one actor forward pass."""

    agents = list(observations)
    batched = {
        key: torch.as_tensor(np.stack([observations[agent][key] for agent in agents]), device=device)
        for key in ACTOR_OBS_KEYS
    }
    logits, next_hidden = policy.forward_actor(batched, hidden)
    decisions = select_factorized_actions(
        logits,
        batched["intent_target_mask"],
        deterministic=deterministic,
    ).cpu().numpy().astype(np.int64)
    return {agent: decisions[index] for index, agent in enumerate(agents)}, next_hidden


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Replace a checkpoint only after a complete file reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _environment_resume_state(env: GhostlineSecurityParallelEnv) -> dict[str, Any]:
    runner = env.runner
    frozen = isinstance(
        runner,
        (FrozenPublishedV1RunnerOpponent, FrozenRunnerV2Opponent),
    )
    runner_kind = (
        f"frozen-{runner_opponent_kind(runner.policy)}"
        if frozen
        else "fair-scripted"
    )
    return {
        "simulation": pickle.dumps(env.sim, protocol=pickle.HIGHEST_PROTOCOL),
        "agents": tuple(env.agents),
        "last_runner_action": int(env._last_runner_action),
        "last_seed": int(env._last_seed),
        "invalid_actions": int(env._invalid_actions),
        "credited_contact_guards": tuple(sorted(env._credited_contact_guards)),
        "runner_kind": runner_kind,
        "runner_id": (
            str(runner.opponent_id)
            if frozen and runner.opponent_id is not None
            else runner_kind
        ),
        "runner_hidden": (
            runner.hidden.detach().cpu().clone()
            if frozen and isinstance(runner.hidden, torch.Tensor)
            else None
        ),
    }


def _restore_environment(
    state: dict[str, Any],
    *,
    frozen_runner_policies: dict[str, Any],
    reward_gamma: float,
) -> tuple[GhostlineSecurityParallelEnv, dict[str, dict[str, np.ndarray]]]:
    simulation = pickle.loads(state["simulation"])
    runner_kind = str(state.get("runner_kind", "fair-scripted"))
    if runner_kind in ("frozen-published-v1", "frozen-runner-v2"):
        runner_id = str(state.get("runner_id", ""))
        frozen_runner_policy = frozen_runner_policies.get(runner_id)
        if frozen_runner_policy is None:
            raise RuntimeError("resume requires the frozen runner checkpoint")
        expected = runner_kind.removeprefix("frozen-")
        actual = runner_opponent_kind(frozen_runner_policy)
        if actual != expected:
            raise RuntimeError(
                f"resume runner kind changed ({actual} != {expected})"
            )
        runner: Any | None = make_frozen_runner_opponent(
            frozen_runner_policy,
            opponent_id=runner_id,
        )
    elif runner_kind == "frozen-v2":
        # Compatibility for an interrupted checkpoint written before the
        # public v1/v2 naming cleanup.
        frozen_runner_policy = next(
            (
                policy
                for policy in frozen_runner_policies.values()
                if runner_opponent_kind(policy) == "published-v1"
            ),
            None,
        )
        if frozen_runner_policy is None:
            raise RuntimeError("resume requires the published-v1 runner checkpoint")
        if runner_opponent_kind(frozen_runner_policy) != "published-v1":
            raise RuntimeError("legacy frozen-v2 resume requires published-v1 policy")
        runner = FrozenPublishedV1RunnerOpponent(
            frozen_runner_policy,
            opponent_id=str(state.get("runner_id", "legacy-published-v1")),
        )
    elif runner_kind == "fair-scripted":
        runner = None
    else:
        raise RuntimeError(f"unknown saved runner kind: {runner_kind}")
    env = GhostlineSecurityParallelEnv(
        tier=int(simulation.tier),
        seed=int(simulation.seed),
        runner=runner,
        reward_gamma=reward_gamma,
    )
    env.sim = simulation
    env.tier = int(simulation.tier)
    env._last_seed = int(state["last_seed"])
    env.agents = list(state["agents"])
    env._last_runner_action = int(state["last_runner_action"])
    env._invalid_actions = int(state["invalid_actions"])
    env._credited_contact_guards = set(
        int(value) for value in state.get("credited_contact_guards", ())
    )
    env._target_cache.clear()
    env._plane_signature = None
    env._plane_cache = None
    if isinstance(
        runner,
        (FrozenPublishedV1RunnerOpponent, FrozenRunnerV2Opponent),
    ):
        runner.reset(simulation)
        saved_hidden = state.get("runner_hidden")
        runner.hidden = saved_hidden.clone() if isinstance(saved_hidden, torch.Tensor) else None
    observations = {agent: env._observation(agent) for agent in env.agents}
    env._current_observations = observations
    return env, observations


def _training_checkpoint(
    policy: SharedSecurityActorCritic,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    steps: int,
    updates: int,
    seed_cursor: int,
    best_worst_tier: float,
    best_selection_key: tuple[float, ...],
    tiers: tuple[int, ...],
    tier_probabilities: np.ndarray,
    args: dict[str, Any],
    rng: np.random.Generator,
    next_validation: int,
    validation_cursor: int,
    resume_state: dict[str, Any] | None = None,
) -> None:
    boundary = {
        "steps": int(steps),
        "updates": int(updates),
        "seed_cursor": int(seed_cursor),
        "next_validation": int(next_validation),
        "validation_cursor": int(validation_cursor),
    }
    if resume_state:
        boundary.update(resume_state)
    _atomic_torch_save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "recurrent_size": policy.recurrent_size,
            "observation_contract": SECURITY_OBSERVATION_CONTRACT,
            "environment_fingerprint": security_environment_fingerprint(),
            "training_source_sha256": _sha256(Path(__file__)),
            "runtime": _runtime_fingerprint(),
            "steps": int(steps),
            "updates": int(updates),
            "seed_cursor": int(seed_cursor),
            "best_worst_tier": float(best_worst_tier),
            "best_selection_key": tuple(float(value) for value in best_selection_key),
            "tiers": tiers,
            "tier_probabilities": tuple(float(value) for value in tier_probabilities),
            "training_args": dict(args),
            "rng_state": rng.bit_generator.state,
            "numpy_global_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "next_validation": int(next_validation),
            "validation_cursor": int(validation_cursor),
            "resume_state": boundary,
        },
        path,
    )


def _tactical_behavior_warmup(
    policy: SharedSecurityActorCritic,
    optimizer: torch.optim.Optimizer,
    envs: list[GhostlineSecurityParallelEnv],
    observations: list[dict[str, dict[str, np.ndarray]]],
    *,
    target_steps: int,
    epochs: int,
    batch_size: int,
    entropy_coefficient: float,
    device: torch.device,
    next_tier: Callable[[], int],
    next_seed: Callable[[], int],
    rng: np.random.Generator,
) -> tuple[list[dict[str, dict[str, np.ndarray]]], dict[str, float | int]]:
    """Imitate the audited tactical baseline before adversarial fine-tuning."""

    storage: dict[str, list[np.ndarray]] = {key: [] for key in ACTOR_OBS_KEYS}
    action_storage: list[np.ndarray] = []
    episodes = runner_successes = 0
    policy.eval()
    while len(action_storage) < target_steps:
        next_records: list[dict[str, dict[str, np.ndarray]]] = []
        for env_index, env in enumerate(envs):
            records = observations[env_index]
            actions: dict[str, np.ndarray] = {}
            for agent, observation in records.items():
                action = tactical_security_action(observation, env.agent_name_mapping[agent])
                actions[agent] = action
                if len(action_storage) < target_steps:
                    for key in ACTOR_OBS_KEYS:
                        storage[key].append(observation[key].copy())
                    action_storage.append(action.copy())
            stepped, _, terminations, truncations, _ = env.step(actions)
            ended = any(terminations.values()) or any(truncations.values())
            if ended:
                episodes += 1
                runner_successes += int(env.sim.extracted)
                tier = next_tier()
                episode_seed = next_seed()
                stepped, _ = env.reset(seed=episode_seed, options={"tier": tier})
            next_records.append(stepped)
        observations = next_records

    dataset = {key: np.stack(values) for key, values in storage.items()}
    actions = np.stack(action_storage)
    final_loss = final_accuracy = final_entropy = 0.0
    policy.train()
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(actions)), math.ceil(len(actions) / batch_size)):
            if len(indices) == 0:
                continue
            tensors = {key: torch.as_tensor(value[indices], device=device) for key, value in dataset.items()}
            expected = torch.as_tensor(actions[indices], device=device)
            logits, _ = policy.forward_actor(tensors)
            log_probability, entropy = factorized_log_prob(
                logits,
                expected,
                tensors["intent_target_mask"],
            )
            loss = -log_probability.mean() - entropy_coefficient * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                predicted = torch.stack([torch.argmax(head, dim=-1) for head in logits], dim=-1)
                final_accuracy = float((predicted == expected).all(dim=-1).float().mean())
            final_loss = float(loss.detach())
            final_entropy = float(entropy.mean().detach())
    return observations, {
        "samples": len(actions),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "final_loss": final_loss,
        "final_exact_action_accuracy": final_accuracy,
        "final_entropy": final_entropy,
        "entropy_coefficient": float(entropy_coefficient),
        "episodes": episodes,
        "runner_success_rate": runner_successes / max(1, episodes),
    }


def evaluate_security_policy(
    policy: SharedSecurityActorCritic | None,
    *,
    tiers: Iterable[int] = (3, 4, 5, 6),
    episodes_per_tier: int = 20,
    seed_start: int = SECURITY_VALIDATION_SEED_START,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    runner_policy: Any | None = None,
    runner_label: str = "fair-scripted",
) -> dict[str, Any]:
    """Evaluate security without exposing centralized critic state to actors."""

    device = torch.device(device)
    records: list[dict[str, Any]] = []
    tier_summaries: dict[str, dict[str, float | int]] = {}
    for tier in parse_security_tiers(tiers):
        stops = 0
        runner_successes = 0
        damage_total = 0
        detections_total = 0
        duration_total = 0.0
        for episode in range(int(episodes_per_tier)):
            seed = int(seed_start + tier * 100_000 + episode)
            runner = (
                make_frozen_runner_opponent(runner_policy)
                if runner_policy is not None
                else None
            )
            env = GhostlineSecurityParallelEnv(tier=tier, seed=seed, runner=runner)
            observations, _ = env.reset(seed=seed)
            hidden_tensor: torch.Tensor | None = None
            while env.agents:
                if policy is None:
                    actions = {
                        agent: tactical_security_action(
                            observations[agent],
                            env.agent_name_mapping[agent],
                        )
                        for agent in env.agents
                    }
                else:
                    actions, hidden_tensor = _batched_security_actions(
                        policy,
                        observations,
                        hidden_tensor,
                        deterministic=deterministic,
                        device=device,
                    )
                observations, _, terminations, truncations, infos = env.step(actions)
                if any(terminations.values()) or any(truncations.values()):
                    break
            info = next(iter(infos.values()))
            runner_success = bool(env.sim.extracted)
            stopped = bool((env.sim.terminated or env.sim.truncated) and not runner_success)
            stops += int(stopped)
            runner_successes += int(runner_success)
            damage_total += int(env.sim.damage_taken)
            detections_total += int(env.sim.detections)
            duration_total += float(env.sim.elapsed_seconds)
            records.append(
                {
                    "tier": tier,
                    "seed": seed,
                    "security_stop": stopped,
                    "runner_success": runner_success,
                    "failure_reason": str(env.sim.fail_reason),
                    "damage": int(env.sim.damage_taken),
                    "detections": int(env.sim.detections),
                    "duration_seconds": float(env.sim.elapsed_seconds),
                    "invalid_actions": int(info["invalid_actions"]),
                }
            )
            env.close()
        count = max(1, int(episodes_per_tier))
        tier_summaries[str(tier)] = {
            "episodes": count,
            "security_stop_rate": stops / count,
            "runner_success_rate": runner_successes / count,
            "mean_damage": damage_total / count,
            "mean_detections": detections_total / count,
            "mean_duration_seconds": duration_total / count,
            "security_stop_ci95_low": _wilson_interval(stops, count)[0],
            "security_stop_ci95_high": _wilson_interval(stops, count)[1],
        }
    worst = min(float(item["security_stop_rate"]) for item in tier_summaries.values())
    return {
        "contract": "ghostline-security-evaluation-v2",
        "observation_contract": SECURITY_OBSERVATION_CONTRACT,
        "environment_fingerprint": security_environment_fingerprint(),
        "seed_start": int(seed_start),
        "episodes_per_tier": int(episodes_per_tier),
        "runner_opponent": runner_label,
        "security_controller": "tactical-observation-only" if policy is None else "recurrent-mappo",
        "tiers": tier_summaries,
        "worst_tier_security_stop_rate": worst,
        "episodes": records,
    }


def train_security(
    *,
    output: Path = Path("artifacts/security-mappo"),
    hours: float = 72.0,
    max_steps: int = 0,
    env_count: int = 8,
    rollout: int = 64,
    epochs: int = 4,
    tiers: str | Iterable[int] = (3, 4, 5, 6),
    recurrent_size: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = SECURITY_REWARD_GAMMA,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 0.5,
    seed: int = 7,
    device: str | None = None,
    validation_interval: int = 100_000,
    validation_episodes: int = 20,
    resume: bool = True,
    dry_run: bool = False,
    runner_checkpoint: Path | None = None,
    runner_pool: Iterable[Path] | None = None,
    init_checkpoint: Path | None = None,
    scripted_opponent_fraction: float = 0.0,
    bc_warmup_steps: int = 0,
    bc_warmup_epochs: int = 2,
    bc_warmup_entropy: float = 0.05,
    adaptive_curriculum: bool = True,
) -> Path:
    selected_tiers = parse_security_tiers(tiers)
    if env_count < 1 or rollout < 2 or epochs < 1:
        raise ValueError("env_count >= 1, rollout >= 2, and epochs >= 1 are required")
    if hours <= 0.0 and max_steps <= 0:
        raise ValueError("hours or max_steps must allow at least one rollout")
    if learning_rate <= 0.0 or not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("learning_rate must be positive and gamma/gae_lambda must be in (0, 1]/[0, 1]")
    if entropy_coefficient < 0.0 or bc_warmup_entropy < 0.0:
        raise ValueError("entropy coefficients cannot be negative")
    if not 0.0 <= scripted_opponent_fraction <= 1.0:
        raise ValueError("scripted_opponent_fraction must be between zero and one")
    if bc_warmup_steps < 0 or bc_warmup_epochs < 1:
        raise ValueError("bc_warmup_steps >= 0 and bc_warmup_epochs >= 1 are required")
    runner_paths: list[Path] = []
    if runner_checkpoint is not None:
        runner_paths.append(Path(runner_checkpoint))
    if runner_pool is not None:
        runner_paths.extend(Path(path) for path in runner_pool)
    resolved_runner_paths: list[Path] = []
    seen_runner_paths: set[Path] = set()
    for path in runner_paths:
        resolved = path.resolve()
        if resolved in seen_runner_paths:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"runner opponent checkpoint is missing: {path}")
        seen_runner_paths.add(resolved)
        resolved_runner_paths.append(resolved)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.pt"
    champion_path = output / "champion.pt"
    last_policy_path = output / "last-policy.pt"
    metrics_path = output / "training-metrics.jsonl"
    manifest_path = output / "experiment-manifest.json"
    existing_outputs = tuple(
        path
        for path in (latest_path, champion_path, last_policy_path)
        if path.exists()
    )
    if not resume and existing_outputs:
        raise RuntimeError(
            f"{output} already contains security checkpoints; "
            "enable resume or choose a new output"
        )

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    training_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frozen_runner_policies: dict[str, Any] = {}
    runner_records: list[dict[str, str]] = []
    runner_label = "fair-scripted"
    for path in resolved_runner_paths:
        policy, kind = load_runner_opponent_policy(
            path,
            device="cpu",
        )
        checkpoint_hash = _sha256(path)
        opponent_id = f"{kind}:{checkpoint_hash}"
        frozen_runner_policies[opponent_id] = policy
        runner_records.append(
            {
                "path": str(path),
                "sha256": checkpoint_hash,
                "kind": kind,
                "opponent_id": opponent_id,
            }
        )
    runner_ids = tuple(sorted(frozen_runner_policies))
    validation_runner_id: str | None = None
    validation_runner_policy: Any | None = None
    if runner_ids:
        digest = hashlib.sha256(
            json.dumps(runner_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        runner_label = f"frozen-pool:{digest}"
        validation_runner_id = next(
            (
                opponent_id
                for opponent_id in reversed(runner_ids)
                if opponent_id.startswith("runner-v2:")
            ),
            runner_ids[-1],
        )
        validation_runner_policy = frozen_runner_policies[
            validation_runner_id
        ]
    policy = SharedSecurityActorCritic(recurrent_size=recurrent_size).to(training_device)
    init_label = None
    if init_checkpoint is not None:
        init_checkpoint = Path(init_checkpoint)
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"security initialization checkpoint is missing: {init_checkpoint}")
        initialized = load_security_policy(init_checkpoint, device=training_device)
        if initialized.recurrent_size != recurrent_size:
            raise RuntimeError("security initialization recurrent size does not match")
        policy.load_state_dict(initialized.state_dict(), strict=True)
        init_label = f"{init_checkpoint}:{_sha256(init_checkpoint)}"
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
    steps = updates = seed_cursor = 0
    best_worst_tier = -math.inf
    best_selection_key: tuple[float, ...] = (-math.inf,) * 6
    tier_probabilities = np.full(len(selected_tiers), 1.0 / len(selected_tiers), dtype=np.float64)
    args_record = {
        "env_count": env_count,
        "rollout": rollout,
        "epochs": epochs,
        "tiers": selected_tiers,
        "recurrent_size": recurrent_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_ratio": clip_ratio,
        "value_coefficient": value_coefficient,
        "entropy_coefficient": entropy_coefficient,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
        "runner_opponent": runner_label,
        "runner_pool": list(runner_ids),
        "init_checkpoint": init_label,
        "scripted_opponent_fraction": float(scripted_opponent_fraction),
        "bc_warmup_steps": int(bc_warmup_steps),
        "bc_warmup_epochs": int(bc_warmup_epochs),
        "bc_warmup_entropy": float(bc_warmup_entropy),
        "adaptive_curriculum": bool(adaptive_curriculum),
        "validation_interval": int(validation_interval),
        "validation_episodes": int(validation_episodes),
        "reward_contract": "bounded-discount-matched-security-v2",
        "device": str(training_device),
    }
    resume_payload: dict[str, Any] | None = None
    next_validation = max(validation_interval, validation_interval)
    validation_cursor = 0
    if resume and latest_path.exists():
        if init_checkpoint is not None:
            raise RuntimeError("cannot combine a resume checkpoint with --init-model")
        payload = torch.load(latest_path, map_location=training_device, weights_only=False)
        if payload.get("environment_fingerprint") != security_environment_fingerprint():
            raise RuntimeError("security resume checkpoint uses a stale environment contract")
        if payload.get("training_source_sha256") != _sha256(Path(__file__)):
            raise RuntimeError("security resume checkpoint uses stale trainer code")
        if payload.get("runtime") != _runtime_fingerprint():
            raise RuntimeError("security resume runtime dependencies do not match")
        if int(payload.get("recurrent_size", recurrent_size)) != recurrent_size:
            raise RuntimeError("security resume recurrent size does not match")
        if tuple(payload.get("tiers", ())) != selected_tiers:
            raise RuntimeError("security resume tier curriculum does not match")
        prior_args = payload.get("training_args")
        if not isinstance(prior_args, dict):
            raise RuntimeError("security resume checkpoint has no training configuration")
        if prior_args.get("runner_opponent") != runner_label:
            raise RuntimeError("security resume runner opponent does not match")
        comparable_prior = dict(prior_args)
        comparable_current = dict(args_record)
        comparable_prior.pop("init_checkpoint", None)
        comparable_current.pop("init_checkpoint", None)
        if comparable_prior != comparable_current:
            raise RuntimeError("security resume training configuration does not match exactly")
        # Preserve initialization provenance across later resume checkpoints;
        # it is metadata, not a hyperparameter that must be supplied again.
        args_record["init_checkpoint"] = prior_args.get("init_checkpoint")
        required_resume_keys = {
            "rng_state",
            "numpy_global_rng_state",
            "python_rng_state",
            "torch_rng_state",
            "next_validation",
            "validation_cursor",
            "resume_state",
        }
        missing = sorted(required_resume_keys - payload.keys())
        if missing:
            raise RuntimeError(f"security resume checkpoint is incomplete: {missing}")
        policy.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        steps = int(payload.get("steps", 0))
        updates = int(payload.get("updates", 0))
        seed_cursor = int(payload.get("seed_cursor", 0))
        best_worst_tier = float(payload.get("best_worst_tier", -math.inf))
        restored_key = tuple(float(value) for value in payload.get("best_selection_key", (best_worst_tier,)))
        best_selection_key = (restored_key + (-math.inf,) * 6)[:6]
        restored_probabilities = np.asarray(payload.get("tier_probabilities", tier_probabilities), dtype=np.float64)
        if restored_probabilities.shape != tier_probabilities.shape or not np.isclose(restored_probabilities.sum(), 1.0):
            raise RuntimeError("security resume tier probabilities are invalid")
        tier_probabilities = restored_probabilities
        next_validation = int(payload["next_validation"])
        validation_cursor = int(payload["validation_cursor"])
        resume_payload = payload

    manifest = {
        "manifest_contract": SECURITY_EXPERIMENT_MANIFEST_CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "preflight-passed"
            if dry_run
            else (
                "training-resumed"
                if resume_payload is not None
                else "training-started"
            )
        ),
        "observation_contract": SECURITY_OBSERVATION_CONTRACT,
        "environment_fingerprint": security_environment_fingerprint(),
        "trainer_sha256": _sha256(Path(__file__)),
        "training_configuration": args_record,
        "budget": {
            "hours": float(hours),
            "max_steps": int(max_steps),
        },
        "initialization": {
            "checkpoint": (
                str(init_checkpoint.resolve())
                if init_checkpoint is not None
                else None
            ),
            "checkpoint_sha256": (
                _sha256(init_checkpoint)
                if init_checkpoint is not None
                else None
            ),
            "behavior_warmup_steps": int(bc_warmup_steps),
        },
        "runner_opponents": runner_records,
        "resume_checkpoint": (
            {
                "path": str(latest_path.resolve()),
                "sha256": _sha256(latest_path),
                "steps": steps,
                "updates": updates,
            }
            if resume_payload is not None
            else None
        ),
        "seed_namespaces": {
            "training_start": SECURITY_TRAIN_SEED_START,
            "validation_start": SECURITY_VALIDATION_SEED_START,
            "final_test_start": SECURITY_FINAL_TEST_SEED_START,
            "final_test_not_consumed_by_training": True,
        },
        "runtime": _runtime_fingerprint(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "selected_device": str(training_device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(training_device)
                if training_device.type == "cuda"
                else None
            ),
        },
        "source": _source_snapshot(),
    }
    if dry_run:
        _atomic_json_save(manifest, manifest_path)
        return manifest_path
    if resume_payload is None or not manifest_path.exists():
        _atomic_json_save(manifest, manifest_path)

    rng = np.random.default_rng(seed)
    if resume_payload is not None:
        rng.bit_generator.state = resume_payload["rng_state"]
    def next_tier() -> int:
        return int(rng.choice(selected_tiers, p=tier_probabilities))

    def next_seed() -> int:
        nonlocal seed_cursor
        value = SECURITY_TRAIN_SEED_START + seed_cursor
        if value >= SECURITY_VALIDATION_SEED_START:
            raise RuntimeError("security training seed namespace exhausted")
        seed_cursor += 1
        return value

    def next_runner() -> (
        FrozenPublishedV1RunnerOpponent
        | FrozenRunnerV2Opponent
        | None
    ):
        if not runner_ids or rng.random() < scripted_opponent_fraction:
            return None
        opponent_id = runner_ids[int(rng.integers(0, len(runner_ids)))]
        return make_frozen_runner_opponent(
            frozen_runner_policies[opponent_id],
            opponent_id=opponent_id,
        )

    envs: list[GhostlineSecurityParallelEnv] = []
    current_observations: list[dict[str, dict[str, np.ndarray]]] = []
    resume_boundary = resume_payload.get("resume_state", {}) if resume_payload is not None else {}
    saved_environments = resume_boundary.get("environments")
    if resume_payload is not None:
        if not isinstance(saved_environments, list) or len(saved_environments) != env_count:
            raise RuntimeError("security resume checkpoint has no exact environment boundary")
        for saved_environment in saved_environments:
            env, observation = _restore_environment(
                saved_environment,
                frozen_runner_policies=frozen_runner_policies,
                reward_gamma=gamma,
            )
            envs.append(env)
            current_observations.append(observation)
    else:
        for _ in range(env_count):
            tier = next_tier()
            episode_seed = next_seed()
            runner = next_runner()
            env = GhostlineSecurityParallelEnv(
                tier=tier,
                seed=episode_seed,
                runner=runner,
                reward_gamma=gamma,
            )
            observation, _ = env.reset(seed=episode_seed, options={"tier": tier})
            envs.append(env)
            current_observations.append(observation)
    started = time.monotonic()
    session_initial_steps = steps
    if steps == 0 and bc_warmup_steps > 0:
        warmup_target = min(int(bc_warmup_steps), max_steps) if max_steps > 0 else int(bc_warmup_steps)
        current_observations, warmup_report = _tactical_behavior_warmup(
            policy,
            optimizer,
            envs,
            current_observations,
            target_steps=warmup_target,
            epochs=bc_warmup_epochs,
            batch_size=256,
            entropy_coefficient=bc_warmup_entropy,
            device=training_device,
            next_tier=next_tier,
            next_seed=next_seed,
            rng=rng,
        )
        steps += warmup_target
        (output / "behavior-warmup.json").write_text(
            json.dumps(warmup_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    current_padded, current_active = _padded_observations(envs, current_observations)
    current_states = np.stack([env.state() for env in envs])
    if resume_payload is not None:
        current_starts = np.asarray(resume_boundary["current_starts"], dtype=bool)
        hidden = torch.as_tensor(
            resume_boundary["hidden"],
            device=training_device,
        ).clone()
        if current_starts.shape != (env_count, MAX_OPERATIVES):
            raise RuntimeError("security resume start mask has the wrong shape")
        if hidden.shape != (1, env_count * MAX_OPERATIVES, recurrent_size):
            raise RuntimeError("security resume hidden state has the wrong shape")
        np.random.set_state(resume_payload["numpy_global_rng_state"])
        random.setstate(resume_payload["python_rng_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        cuda_state = resume_payload.get("cuda_rng_state")
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    else:
        current_starts = np.ones((env_count, MAX_OPERATIVES), dtype=bool)
        hidden = torch.zeros(1, env_count * MAX_OPERATIVES, recurrent_size, device=training_device)
    deadline = started + hours * 3600.0
    if resume_payload is None:
        next_validation = max(validation_interval, steps + validation_interval)

    try:
        while time.monotonic() < deadline and (max_steps <= 0 or steps < max_steps):
            rollout_initial_hidden = hidden.detach().clone()
            observation_buffer = {key: [] for key in ACTOR_OBS_KEYS}
            state_buffer: list[np.ndarray] = []
            active_buffer: list[np.ndarray] = []
            start_buffer: list[np.ndarray] = []
            action_buffer: list[np.ndarray] = []
            log_probability_buffer: list[np.ndarray] = []
            value_buffer: list[np.ndarray] = []
            reward_buffer: list[np.ndarray] = []
            agent_reward_buffer: list[np.ndarray] = []
            done_buffer: list[np.ndarray] = []
            episodes_finished = runner_successes = 0
            reward_component_sums: dict[str, float] = {}
            reward_component_count = 0

            policy.eval()
            for _ in range(rollout):
                for key in ACTOR_OBS_KEYS:
                    observation_buffer[key].append(current_padded[key].copy())
                state_buffer.append(current_states.copy())
                active_buffer.append(current_active.copy())
                start_buffer.append(current_starts.copy())
                tensors = _actor_tensors(current_padded, training_device)
                state_tensor = torch.as_tensor(current_states, device=training_device)
                with torch.no_grad():
                    logits, next_hidden = policy.forward_actor(tensors, hidden)
                    sampled = _sample_actions(
                        logits,
                        tensors["intent_target_mask"],
                    )
                    log_probability, _ = factorized_log_prob(
                        logits,
                        sampled,
                        tensors["intent_target_mask"],
                    )
                    values = policy.value(state_tensor)
                sampled_np = sampled.reshape(env_count, MAX_OPERATIVES, 4).cpu().numpy()
                rewards = np.zeros(env_count, dtype=np.float32)
                agent_rewards = np.zeros((env_count, MAX_OPERATIVES), dtype=np.float32)
                dones = np.zeros(env_count, dtype=bool)
                next_records: list[dict[str, dict[str, np.ndarray]]] = []
                next_starts = np.zeros((env_count, MAX_OPERATIVES), dtype=bool)
                for env_index, env in enumerate(envs):
                    actions = {
                        agent: sampled_np[env_index, env.agent_name_mapping[agent]]
                        for agent in env.agents
                    }
                    observations, team_rewards, terminations, truncations, infos = env.step(actions)
                    # The critic predicts the shared team value, so its target
                    # uses the shared component; the actor is credited with the
                    # operative's own containment shaping on top of it.
                    for agent, agent_reward in team_rewards.items():
                        agent_rewards[env_index, env.agent_name_mapping[agent]] = float(agent_reward)
                    if infos:
                        components = next(iter(infos.values())).get("reward_components", {})
                        rewards[env_index] = float(components.get("total", next(iter(team_rewards.values()))))
                        for name, value in components.items():
                            if name != "total":
                                reward_component_sums[name] = reward_component_sums.get(name, 0.0) + float(value)
                        reward_component_count += 1
                    ended = any(terminations.values()) or any(truncations.values())
                    dones[env_index] = ended
                    if ended:
                        episodes_finished += 1
                        runner_successes += int(env.sim.extracted)
                        tier = next_tier()
                        episode_seed = next_seed()
                        selected_runner = next_runner()
                        env.runner = env._scripted_runner.act if selected_runner is None else selected_runner
                        observations, _ = env.reset(seed=episode_seed, options={"tier": tier})
                        next_starts[env_index] = True
                    next_records.append(observations)
                action_buffer.append(sampled_np)
                log_probability_buffer.append(log_probability.reshape(env_count, MAX_OPERATIVES).cpu().numpy())
                value_buffer.append(values.cpu().numpy())
                reward_buffer.append(rewards)
                agent_reward_buffer.append(agent_rewards)
                done_buffer.append(dones)
                hidden = next_hidden.detach()
                for env_index, ended in enumerate(dones):
                    if ended:
                        slots = slice(env_index * MAX_OPERATIVES, (env_index + 1) * MAX_OPERATIVES)
                        hidden[:, slots, :] = 0.0
                current_observations = next_records
                current_padded, current_active = _padded_observations(envs, current_observations)
                current_states = np.stack([env.state() for env in envs])
                current_starts = next_starts
                steps += int(active_buffer[-1].sum())
                if max_steps > 0 and steps >= max_steps:
                    break

            actual_rollout = len(reward_buffer)
            with torch.no_grad():
                next_values = policy.value(
                    torch.as_tensor(current_states, device=training_device)
                ).cpu().numpy()
            rewards_np = np.stack(reward_buffer)
            agent_rewards_np = np.stack(agent_reward_buffer)
            dones_np = np.stack(done_buffer)
            values_np = np.stack(value_buffer)
            active_np = np.stack(active_buffer).astype(np.float32)

            # Each operative gets V_i(global state, own shared-encoder block).
            # GAE and all statistics are then masked by actual presence; padded
            # slots contribute neither targets nor gradients.
            agent_advantages = np.zeros_like(agent_rewards_np)
            last_agent_advantage = np.zeros((env_count, MAX_OPERATIVES), dtype=np.float32)
            for index in reversed(range(actual_rollout)):
                continuation = (1.0 - dones_np[index].astype(np.float32))[:, None]
                following = next_values if index == actual_rollout - 1 else values_np[index + 1]
                delta = (
                    agent_rewards_np[index]
                    + gamma * following * continuation
                    - values_np[index]
                )
                last_agent_advantage = (
                    delta
                    + gamma * gae_lambda * continuation * last_agent_advantage
                ) * active_np[index]
                agent_advantages[index] = last_agent_advantage
            returns = (agent_advantages + values_np) * active_np
            normalized_advantages = _normalize_active_advantages(
                agent_advantages,
                active_np,
            )

            sequence_observation = {
                key: torch.as_tensor(np.stack(values), device=training_device).flatten(1, 2)
                for key, values in observation_buffer.items()
            }
            actions_tensor = torch.as_tensor(np.stack(action_buffer), device=training_device).flatten(1, 2)
            old_log_probability = torch.as_tensor(np.stack(log_probability_buffer), device=training_device).flatten(1, 2)
            active_tensor = torch.as_tensor(np.stack(active_buffer), device=training_device).flatten(1, 2)
            reset_tensor = torch.as_tensor(np.stack(start_buffer), device=training_device).flatten(1, 2)
            actor_advantage = torch.as_tensor(normalized_advantages, device=training_device).flatten(1, 2)
            returns_tensor = torch.as_tensor(returns, device=training_device)
            states_tensor = torch.as_tensor(np.stack(state_buffer), device=training_device)
            old_values_tensor = torch.as_tensor(values_np, device=training_device)
            value_active_tensor = torch.as_tensor(active_np, device=training_device)

            policy.train()
            final_policy_loss = final_value_loss = final_entropy = final_clip_fraction = 0.0
            for _ in range(epochs):
                logits, _ = policy.forward_actor_sequence(
                    sequence_observation,
                    rollout_initial_hidden,
                    reset_tensor,
                )
                new_log_probability, entropy = factorized_log_prob(
                    logits,
                    actions_tensor,
                    sequence_observation["intent_target_mask"],
                )
                ratio = torch.exp(new_log_probability - old_log_probability)
                unclipped = ratio * actor_advantage
                clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * actor_advantage
                policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), active_tensor)
                predicted_values = policy.value(states_tensor)
                value_loss = _masked_value_loss(
                    predicted_values,
                    old_values_tensor,
                    returns_tensor,
                    value_active_tensor,
                    clip_ratio,
                )
                entropy_mean = _masked_mean(entropy, active_tensor)
                loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()
                final_policy_loss = float(policy_loss.detach())
                final_value_loss = float(value_loss.detach())
                final_entropy = float(entropy_mean.detach())
                final_clip_fraction = float(_masked_mean(((ratio - 1.0).abs() > clip_ratio).float(), active_tensor).detach())
            updates += 1
            elapsed = max(1e-6, time.monotonic() - started)
            sampled_actions = np.stack(action_buffer)
            active_actions = np.stack(active_buffer).astype(bool)
            action_histograms = {
                name: np.bincount(sampled_actions[..., index][active_actions], minlength=size).tolist()
                for index, (name, size) in enumerate(
                    zip(SECURITY_ACTION_FACTORS, SECURITY_ACTION_SIZES, strict=True)
                )
            }
            record = {
                "update": updates,
                "steps": steps,
                "decisions_per_second": (steps - session_initial_steps) / elapsed,
                "policy_loss": final_policy_loss,
                "value_loss": final_value_loss,
                "entropy": final_entropy,
                "clip_fraction": final_clip_fraction,
                "mean_team_reward": float(rewards_np.mean()),
                "episodes_finished": episodes_finished,
                "runner_success_rate": runner_successes / max(1, episodes_finished),
                "action_histograms": action_histograms,
                "mean_reward_components": {
                    name: value / max(1, reward_component_count)
                    for name, value in sorted(reward_component_sums.items())
                },
                "tier_probabilities": {
                    str(tier): float(probability)
                    for tier, probability in zip(selected_tiers, tier_probabilities, strict=True)
                },
            }
            if validation_interval > 0 and steps >= next_validation:
                validation_seed = (
                    SECURITY_VALIDATION_SEED_START
                    + validation_cursor * max(1, validation_episodes)
                )
                if (
                    validation_seed
                    + max(selected_tiers) * 100_000
                    + validation_episodes
                    >= SECURITY_FINAL_TEST_SEED_START
                ):
                    raise RuntimeError("security validation seed namespace exhausted")
                policy.eval()
                report = evaluate_security_policy(
                    policy,
                    tiers=selected_tiers,
                    episodes_per_tier=validation_episodes,
                    seed_start=validation_seed,
                    device=training_device,
                    runner_policy=validation_runner_policy,
                    runner_label=(
                        validation_runner_id
                        if validation_runner_id is not None
                        else "fair-scripted"
                    ),
                )
                (output / f"validation-{steps:012d}.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                selection_key = _selection_key(report)
                save_security_policy(
                    policy,
                    output / f"policy-{steps:012d}.pt",
                    steps=steps,
                    updates=updates,
                    runner_opponent=runner_label,
                    selection_key=selection_key,
                    validation=report["tiers"],
                    purpose="immutable_validation_checkpoint",
                )
                if selection_key > best_selection_key:
                    best_selection_key = selection_key
                    best_worst_tier = selection_key[0]
                    save_security_policy(
                        policy,
                        champion_path,
                        steps=steps,
                        updates=updates,
                        runner_opponent=runner_label,
                        selection_key=selection_key,
                        validation=report["tiers"],
                    )
                if adaptive_curriculum:
                    tier_probabilities = _adaptive_tier_probabilities(report, selected_tiers)
                validation_cursor += 1
                next_validation = steps + validation_interval
            resume_state = {
                "environments": [_environment_resume_state(env) for env in envs],
                "current_starts": current_starts.copy(),
                "hidden": hidden.detach().cpu().clone(),
            }
            _training_checkpoint(
                policy,
                optimizer,
                latest_path,
                steps=steps,
                updates=updates,
                seed_cursor=seed_cursor,
                best_worst_tier=best_worst_tier,
                best_selection_key=best_selection_key,
                tiers=selected_tiers,
                tier_probabilities=tier_probabilities,
                args=args_record,
                rng=rng,
                next_validation=next_validation,
                validation_cursor=validation_cursor,
                resume_state=resume_state,
            )
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        for env in envs:
            env.close()

    save_security_policy(
        policy,
        last_policy_path,
        steps=steps,
        updates=updates,
        runner_opponent=runner_label,
        purpose="security-v2-last-policy",
        validated_champion=champion_path.exists(),
    )
    return (
        champion_path
        if champion_path.exists()
        else last_policy_path
    )


def evaluate_security_checkpoint(
    *,
    model: Path | None,
    output: Path,
    tiers: str | Iterable[int] = (3, 4, 5, 6),
    episodes_per_tier: int = 100,
    seed_start: int = SECURITY_FINAL_TEST_SEED_START,
    device: str | None = None,
    runner_checkpoint: Path | None = None,
    slice_manifest: Path | None = None,
) -> Path:
    output = Path(output)
    if output.suffix.casefold() != ".json":
        raise ValueError("security evaluation output must use a .json suffix")
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; reserve a new slice and output"
        )
    if episodes_per_tier <= 0:
        raise ValueError("episodes_per_tier must be positive")
    selected_tiers = parse_security_tiers(tiers)
    model_path = Path(model).expanduser().resolve() if model is not None else None
    if model_path is not None and not model_path.is_file():
        raise FileNotFoundError(f"security checkpoint is missing: {model_path}")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = (
        load_security_policy(model_path, device=selected_device)
        if model_path is not None
        else None
    )
    runner_policy = None
    runner_label = "fair-scripted"
    if runner_checkpoint is not None:
        runner_checkpoint = Path(runner_checkpoint).expanduser().resolve()
        if not runner_checkpoint.is_file():
            raise FileNotFoundError(
                f"runner opponent checkpoint is missing: {runner_checkpoint}"
            )
        runner_policy, kind = load_runner_opponent_policy(
            runner_checkpoint,
            device="cpu",
        )
        runner_label = f"{kind}:{_sha256(runner_checkpoint)}"
    checkpoint_sha256 = _sha256(model_path) if model_path is not None else None
    policy_kind = (
        "security-v2-neural" if model_path is not None else "security-v2-tactical"
    ) + f"-vs-{runner_label}"
    lease = None
    if slice_manifest is not None:
        if seed_start < SECURITY_FINAL_TEST_SEED_START:
            raise ValueError(
                "security final evaluation seed_start must be at least "
                f"{SECURITY_FINAL_TEST_SEED_START:,}"
            )
        from ghostline.evaluation import _open_final_slice

        lease = _open_final_slice(
            manifest_path=Path(slice_manifest),
            seed_start=seed_start,
            episodes=episodes_per_tier,
            tiers=selected_tiers,
            environment_fingerprint=security_environment_fingerprint(),
            policy_kind=policy_kind,
            checkpoint_sha256=checkpoint_sha256,
            output=output,
            observation_contract=SECURITY_OBSERVATION_CONTRACT,
        )
    try:
        report = evaluate_security_policy(
            policy,
            tiers=selected_tiers,
            episodes_per_tier=episodes_per_tier,
            seed_start=seed_start,
            device=selected_device,
            runner_policy=runner_policy,
            runner_label=runner_label,
        )
        report["security_checkpoint_sha256"] = checkpoint_sha256
        report["release_audit"] = lease is not None
        report["slice_manifest"] = (
            Path(slice_manifest).as_posix() if slice_manifest is not None else None
        )
        report["slice_audit_id"] = lease.audit_id if lease is not None else None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        aggregate_path = output.with_suffix(".csv")
        with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "tier",
                    "episodes",
                    "security_stop_rate",
                    "security_stop_ci95_low",
                    "security_stop_ci95_high",
                    "runner_success_rate",
                    "mean_damage",
                    "mean_detections",
                    "mean_duration_seconds",
                ),
            )
            writer.writeheader()
            for tier, summary in report["tiers"].items():
                writer.writerow({"tier": tier, **summary})
        episode_path = output.with_name(f"{output.stem}.episodes.csv")
        with episode_path.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = (
                tuple(report["episodes"][0])
                if report["episodes"]
                else ("tier", "seed")
            )
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report["episodes"])
        if lease is not None:
            lease.finalize(
                report,
                (output, aggregate_path, episode_path),
            )
        return output
    except BaseException as error:
        if lease is not None:
            lease.abort(error)
        raise
