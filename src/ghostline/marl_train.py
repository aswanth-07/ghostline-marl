"""Parameter-shared recurrent MAPPO for Ghostline's adaptive security team."""
from __future__ import annotations

import csv
import json
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn

from ghostline.config_v3 import MAX_SECURITY_TARGETS
from ghostline.security_env import GhostlineSecurityParallelEnv
from ghostline.runner_opponents import FrozenV2RunnerOpponent
from ghostline.security_baselines import tactical_security_action
from ghostline.model import load_policy
from ghostline.security_model import (
    SECURITY_MASK_KEYS,
    SECURITY_OBSERVATION_CONTRACT,
    SharedSecurityActorCritic,
    factorized_log_prob,
    load_security_policy,
    save_security_policy,
    security_environment_fingerprint,
)


SECURITY_TRAIN_SEED_START = 10_000_000
SECURITY_VALIDATION_SEED_START = 11_000_000
SECURITY_FINAL_TEST_SEED_START = 12_000_000
MAX_OPERATIVES = 5
ACTOR_OBS_KEYS = (
    "ego",
    "local_grid",
    "runner",
    "teammates",
    "teammate_mask",
    "targets",
    "target_mask",
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


def _sample_actions(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    return torch.stack(
        [torch.distributions.Categorical(logits=head).sample() for head in logits],
        dim=-1,
    )


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
    if deterministic:
        factors = [torch.argmax(head, dim=-1) for head in logits]
    else:
        factors = [torch.distributions.Categorical(logits=head).sample() for head in logits]
    decisions = torch.stack(factors, dim=-1).cpu().numpy().astype(np.int64)
    return {agent: decisions[index] for index, agent in enumerate(agents)}, next_hidden


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "recurrent_size": policy.recurrent_size,
            "observation_contract": SECURITY_OBSERVATION_CONTRACT,
            "environment_fingerprint": security_environment_fingerprint(),
            "steps": int(steps),
            "updates": int(updates),
            "seed_cursor": int(seed_cursor),
            "best_worst_tier": float(best_worst_tier),
            "best_selection_key": tuple(float(value) for value in best_selection_key),
            "tiers": tiers,
            "tier_probabilities": tuple(float(value) for value in tier_probabilities),
            "training_args": args,
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
            log_probability, entropy = factorized_log_prob(logits, expected)
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
            runner = FrozenV2RunnerOpponent(runner_policy) if runner_policy is not None else None
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
        "contract": "ghostline-security-evaluation-v0",
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
    gamma: float = 0.995,
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.pt"
    champion_path = output / "champion.pt"
    metrics_path = output / "training-metrics.jsonl"
    if dry_run:
        manifest = {
            "contract": SECURITY_OBSERVATION_CONTRACT,
            "environment_fingerprint": security_environment_fingerprint(),
            "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "tiers": selected_tiers,
            "env_count": env_count,
            "rollout": rollout,
            "hours": hours,
            "max_steps": max_steps,
            "runner_checkpoint": str(runner_checkpoint) if runner_checkpoint is not None else None,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
            "scripted_opponent_fraction": float(scripted_opponent_fraction),
            "bc_warmup_steps": int(bc_warmup_steps),
            "bc_warmup_epochs": int(bc_warmup_epochs),
            "bc_warmup_entropy": float(bc_warmup_entropy),
            "adaptive_curriculum": bool(adaptive_curriculum),
        }
        (output / "dry-run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output / "dry-run.json"

    torch.manual_seed(seed)
    np.random.seed(seed)
    training_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frozen_runner_policy = None
    runner_label = "fair-scripted"
    if runner_checkpoint is not None:
        runner_checkpoint = Path(runner_checkpoint)
        if not runner_checkpoint.is_file():
            raise FileNotFoundError(f"runner opponent checkpoint is missing: {runner_checkpoint}")
        frozen_runner_policy = load_policy(runner_checkpoint, device="cpu")
        runner_label = f"env-v2:{_sha256(runner_checkpoint)}"
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
    if resume and latest_path.exists():
        if init_checkpoint is not None:
            raise RuntimeError("cannot combine a resume checkpoint with --init-model")
        payload = torch.load(latest_path, map_location=training_device, weights_only=False)
        if payload.get("environment_fingerprint") != security_environment_fingerprint():
            raise RuntimeError("security resume checkpoint uses a stale environment contract")
        if int(payload.get("recurrent_size", recurrent_size)) != recurrent_size:
            raise RuntimeError("security resume recurrent size does not match")
        if tuple(payload.get("tiers", ())) != selected_tiers:
            raise RuntimeError("security resume tier curriculum does not match")
        prior_runner = payload.get("training_args", {}).get("runner_opponent")
        if prior_runner != runner_label:
            raise RuntimeError("security resume runner opponent does not match")
        prior_adaptive = bool(payload.get("training_args", {}).get("adaptive_curriculum", False))
        if prior_adaptive != adaptive_curriculum:
            raise RuntimeError("security resume curriculum mode does not match")
        prior_scripted_fraction = float(payload.get("training_args", {}).get("scripted_opponent_fraction", 0.0))
        if not math.isclose(prior_scripted_fraction, scripted_opponent_fraction):
            raise RuntimeError("security resume opponent curriculum does not match")
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

    rng = np.random.default_rng(seed + seed_cursor)
    def next_tier() -> int:
        return int(rng.choice(selected_tiers, p=tier_probabilities))

    def next_seed() -> int:
        nonlocal seed_cursor
        value = SECURITY_TRAIN_SEED_START + seed_cursor
        seed_cursor += 1
        return value

    def next_runner() -> FrozenV2RunnerOpponent | None:
        if frozen_runner_policy is None or rng.random() < scripted_opponent_fraction:
            return None
        return FrozenV2RunnerOpponent(frozen_runner_policy)

    envs: list[GhostlineSecurityParallelEnv] = []
    current_observations: list[dict[str, dict[str, np.ndarray]]] = []
    for _ in range(env_count):
        tier = next_tier()
        episode_seed = next_seed()
        runner = next_runner()
        env = GhostlineSecurityParallelEnv(tier=tier, seed=episode_seed, runner=runner)
        observation, _ = env.reset(seed=episode_seed, options={"tier": tier})
        envs.append(env)
        current_observations.append(observation)
    started = time.monotonic()
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
    current_starts = np.ones((env_count, MAX_OPERATIVES), dtype=bool)
    hidden = torch.zeros(1, env_count * MAX_OPERATIVES, recurrent_size, device=training_device)
    deadline = started + hours * 3600.0
    next_validation = max(validation_interval, steps + validation_interval)
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
        "seed": seed,
        "runner_opponent": runner_label,
        "init_checkpoint": init_label,
        "scripted_opponent_fraction": float(scripted_opponent_fraction),
        "bc_warmup_steps": int(bc_warmup_steps),
        "bc_warmup_epochs": int(bc_warmup_epochs),
        "bc_warmup_entropy": float(bc_warmup_entropy),
        "adaptive_curriculum": bool(adaptive_curriculum),
    }

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
                    sampled = _sample_actions(logits)
                    log_probability, _ = factorized_log_prob(logits, sampled)
                    values = policy.value(state_tensor)
                sampled_np = sampled.reshape(env_count, MAX_OPERATIVES, 4).cpu().numpy()
                rewards = np.zeros(env_count, dtype=np.float32)
                dones = np.zeros(env_count, dtype=bool)
                next_records: list[dict[str, dict[str, np.ndarray]]] = []
                next_starts = np.zeros((env_count, MAX_OPERATIVES), dtype=bool)
                for env_index, env in enumerate(envs):
                    actions = {
                        agent: sampled_np[env_index, env.agent_name_mapping[agent]]
                        for agent in env.agents
                    }
                    observations, team_rewards, terminations, truncations, infos = env.step(actions)
                    rewards[env_index] = float(next(iter(team_rewards.values())))
                    if infos:
                        components = next(iter(infos.values())).get("reward_components", {})
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
                next_values = policy.value(torch.as_tensor(current_states, device=training_device)).cpu().numpy()
            rewards_np = np.stack(reward_buffer)
            dones_np = np.stack(done_buffer)
            values_np = np.stack(value_buffer)
            advantages = np.zeros_like(rewards_np)
            last_advantage = np.zeros(env_count, dtype=np.float32)
            for index in reversed(range(actual_rollout)):
                continuation = 1.0 - dones_np[index].astype(np.float32)
                following = next_values if index == actual_rollout - 1 else values_np[index + 1]
                delta = rewards_np[index] + gamma * following * continuation - values_np[index]
                last_advantage = delta + gamma * gae_lambda * continuation * last_advantage
                advantages[index] = last_advantage
            returns = advantages + values_np
            normalized_advantages = (advantages - advantages.mean()) / max(1e-6, advantages.std())

            sequence_observation = {
                key: torch.as_tensor(np.stack(values), device=training_device).flatten(1, 2)
                for key, values in observation_buffer.items()
            }
            actions_tensor = torch.as_tensor(np.stack(action_buffer), device=training_device).flatten(1, 2)
            old_log_probability = torch.as_tensor(np.stack(log_probability_buffer), device=training_device).flatten(1, 2)
            active_tensor = torch.as_tensor(np.stack(active_buffer), device=training_device).flatten(1, 2)
            reset_tensor = torch.as_tensor(np.stack(start_buffer), device=training_device).flatten(1, 2)
            advantage_tensor = torch.as_tensor(normalized_advantages, device=training_device)
            actor_advantage = advantage_tensor.unsqueeze(-1).expand(-1, -1, MAX_OPERATIVES).flatten(1, 2)
            returns_tensor = torch.as_tensor(returns, device=training_device)
            states_tensor = torch.as_tensor(np.stack(state_buffer), device=training_device)
            old_values_tensor = torch.as_tensor(values_np, device=training_device)

            policy.train()
            final_policy_loss = final_value_loss = final_entropy = final_clip_fraction = 0.0
            for _ in range(epochs):
                logits, _ = policy.forward_actor_sequence(
                    sequence_observation,
                    rollout_initial_hidden,
                    reset_tensor,
                )
                new_log_probability, entropy = factorized_log_prob(logits, actions_tensor)
                ratio = torch.exp(new_log_probability - old_log_probability)
                unclipped = ratio * actor_advantage
                clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * actor_advantage
                policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), active_tensor)
                predicted_values = policy.value(states_tensor)
                clipped_values = old_values_tensor + (predicted_values - old_values_tensor).clamp(-clip_ratio, clip_ratio)
                value_loss = 0.5 * torch.maximum(
                    (predicted_values - returns_tensor).square(),
                    (clipped_values - returns_tensor).square(),
                ).mean()
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
                    (("intent", 8), ("target", MAX_SECURITY_TARGETS), ("message", 5), ("ability", 2))
                )
            }
            record = {
                "update": updates,
                "steps": steps,
                "decisions_per_second": steps / elapsed,
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
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
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
            )

            if validation_interval > 0 and steps >= next_validation:
                policy.eval()
                report = evaluate_security_policy(
                    policy,
                    tiers=selected_tiers,
                    episodes_per_tier=validation_episodes,
                    seed_start=SECURITY_VALIDATION_SEED_START,
                    device=training_device,
                    runner_policy=frozen_runner_policy,
                    runner_label=runner_label,
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
                next_validation = steps + validation_interval
    finally:
        for env in envs:
            env.close()

    if not champion_path.exists():
        save_security_policy(
            policy,
            champion_path,
            steps=steps,
            updates=updates,
            runner_opponent=runner_label,
            selection="last_without_validation",
        )
    return champion_path


def evaluate_security_checkpoint(
    *,
    model: Path | None,
    output: Path,
    tiers: str | Iterable[int] = (3, 4, 5, 6),
    episodes_per_tier: int = 100,
    seed_start: int = SECURITY_FINAL_TEST_SEED_START,
    device: str | None = None,
    runner_checkpoint: Path | None = None,
) -> Path:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_security_policy(Path(model), device=selected_device) if model is not None else None
    runner_policy = None
    runner_label = "fair-scripted"
    if runner_checkpoint is not None:
        runner_checkpoint = Path(runner_checkpoint)
        runner_policy = load_policy(runner_checkpoint, device="cpu")
        runner_label = f"env-v2:{_sha256(runner_checkpoint)}"
    report = evaluate_security_policy(
        policy,
        tiers=parse_security_tiers(tiers),
        episodes_per_tier=episodes_per_tier,
        seed_start=seed_start,
        device=selected_device,
        runner_policy=runner_policy,
        runner_label=runner_label,
    )
    report["security_checkpoint_sha256"] = _sha256(Path(model)) if model is not None else None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        fieldnames = tuple(report["episodes"][0]) if report["episodes"] else ("tier", "seed")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["episodes"])
    return output
