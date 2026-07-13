"""Recurrent clipped-PPO fine-tuning with curriculum replay and decaying RND."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ghostline.curriculum import AdaptiveCurriculum
from ghostline.env import GhostlineEnv
from ghostline.imitation import OBS_KEYS
from ghostline.model import (
    OBSERVATION_CONTRACT,
    UniversalGhostlinePolicy,
    current_environment_fingerprint,
    load_policy,
    require_current_checkpoint,
    save_policy,
)
from ghostline.rnd import RandomNetworkDistillation, decaying_rnd_coefficient
from ghostline.seeds import VALIDATION_TIER_STRIDE, validation_seed


FULL_VALIDATION_TIERS = frozenset(range(1, 7))
INITIAL_ROLLBACK_FILENAME = "initial-rollback.pt"
RESUME_CONTRACT_FIELDS = (
    "seed",
    "envs",
    "rollout",
    "epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_ratio",
    "value_coef",
    "entropy_coef",
    "max_grad_norm",
    "rnd_coef",
    "rnd_learning_rate",
    "rnd_decay_steps",
    "rnd_final_fraction",
    "rnd_update_fraction",
    "anchor_kl_coef",
    "anchor_decay_steps",
    "anchor_final_fraction",
    "recurrent_size",
    "feedforward",
    "cpu",
    "async_envs",
    "curriculum",
    "fixed_tier",
    "training_lesson",
    "validation_interval",
    "validation_episodes",
    "initial_validation_cursor",
    "initial_curriculum_tier",
)


class CurriculumEnv(GhostlineEnv):
    def __init__(
        self,
        rank: int,
        curriculum: AdaptiveCurriculum,
        training_lesson: int = 0,
        fixed_tier: int | None = None,
    ):
        self._curriculum = curriculum
        self._curriculum_rng = np.random.default_rng(rank * 1009 + 17)
        self._training_lesson = training_lesson
        self._fixed_tier = fixed_tier
        super().__init__(seed=rank * 1009, tier=1)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault(
            "tier",
            self._fixed_tier
            if self._fixed_tier is not None
            else self._curriculum.sample_tier(self._curriculum_rng),
        )
        if self._training_lesson:
            options.setdefault("training_lesson", self._training_lesson)
        return super().reset(seed=seed, options=options)


def _make_env(rank: int, curriculum: AdaptiveCurriculum, training_lesson: int, fixed_tier: int | None):
    return lambda: CurriculumEnv(rank, curriculum, training_lesson, fixed_tier)


def make_curriculum_vector_env(
    *,
    env_count: int,
    curriculum: AdaptiveCurriculum,
    training_lesson: int,
    fixed_tier: int | None,
    async_envs: bool,
):
    """Build a vector environment without recurrent-state-contaminating reset ticks.

    Gymnasium's default ``NEXT_STEP`` autoreset inserts an ignored-action
    transition after every terminal state. For a recurrent policy that both
    trains on a fictitious zero-reward transition and carries terminal-state
    memory into the reset observation. ``SAME_STEP`` returns the new initial
    observation together with ``final_info`` on the terminal transition, which
    matches the hidden-state reset already performed by the learner.
    """
    vector_class = gym.vector.AsyncVectorEnv if async_envs else gym.vector.SyncVectorEnv
    return vector_class(
        [
            _make_env(rank, curriculum, training_lesson, fixed_tier)
            for rank in range(env_count)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def _tensor_obs(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(obs[key], device=device) for key in OBS_KEYS}


def completed_episode_successes(infos: dict, done: np.ndarray) -> list[float]:
    """Extract terminal success from current and legacy vector-info layouts."""

    final_info = infos.get("final_info")
    final_mask = np.asarray(infos.get("_final_info", done), dtype=bool)
    results: list[float] = []
    if isinstance(final_info, dict):
        values = np.asarray(final_info.get("is_success", np.zeros_like(done, dtype=bool)))
        value_mask = np.asarray(final_info.get("_is_success", final_mask), dtype=bool)
        for index, ended in enumerate(np.asarray(done, dtype=bool)):
            if ended and final_mask[index] and value_mask[index]:
                results.append(float(values[index]))
        return results
    if final_info is None:
        return results
    for index, ended in enumerate(np.asarray(done, dtype=bool)):
        if not ended or index >= len(final_info) or not final_mask[index]:
            continue
        item = final_info[index]
        if item:
            results.append(float(item.get("is_success", False)))
    return results


def mask_terminal_intrinsic_rewards(
    intrinsic: torch.Tensor,
    done: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Do not award reset-state novelty on a same-step terminal transition."""

    terminal = torch.as_tensor(done, dtype=torch.bool, device=intrinsic.device)
    if terminal.shape != intrinsic.shape:
        raise ValueError("terminal mask must match the intrinsic reward batch")
    return intrinsic.masked_fill(terminal, 0.0)


def categorical_anchor_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(reference || current) for a conservative imitation-policy anchor."""

    if current_logits.shape != reference_logits.shape:
        raise ValueError("anchor logits must have identical shapes")
    reference_log_probability = torch.log_softmax(reference_logits, dim=-1)
    current_log_probability = torch.log_softmax(current_logits, dim=-1)
    reference_probability = reference_log_probability.exp()
    return torch.sum(
        reference_probability * (reference_log_probability - current_log_probability),
        dim=-1,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_initial_rollback(
    policy: UniversalGhostlinePolicy,
    destination: Path,
    *,
    source: Path | None,
    allow_existing: bool,
) -> tuple[str, bool]:
    """Create once, then verify, the policy used for PPO rollback and KL anchoring."""

    source_sha256 = _sha256(source) if source is not None else None
    if destination.exists():
        if not allow_existing:
            raise RuntimeError(
                f"PPO experiment already owns {destination}; resume it or choose a new experiment name"
            )
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        require_current_checkpoint(payload, path=destination)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if metadata.get("selection_stage") != "immutable_initial_rollback":
            raise RuntimeError("PPO initial rollback checkpoint has an invalid provenance stage")
        recorded_source = metadata.get("source_checkpoint_sha256")
        if source_sha256 is not None and recorded_source != source_sha256:
            raise RuntimeError("PPO initialization checkpoint differs from the immutable rollback source")
        return _sha256(destination), recorded_source is not None

    if allow_existing:
        raise RuntimeError("Cannot resume PPO because its immutable initial rollback checkpoint is missing")
    save_policy(
        policy,
        destination,
        selection_stage="immutable_initial_rollback",
        source_checkpoint_sha256=source_sha256,
    )
    return _sha256(destination), source_sha256 is not None


def _resume_contract(args: argparse.Namespace, rollback_sha256: str) -> dict[str, object]:
    contract = {name: getattr(args, name) for name in RESUME_CONTRACT_FIELDS}
    contract["rollback_sha256"] = rollback_sha256
    return contract


def _require_matching_resume_contract(
    checkpoint: dict[str, object],
    expected: dict[str, object],
) -> None:
    actual = checkpoint.get("resume_contract")
    if not isinstance(actual, dict):
        raise RuntimeError("PPO resume state predates the fail-closed resume contract")
    mismatches = {
        name: (actual.get(name), value)
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"PPO resume contract changed: {mismatches}")


def require_validation_window(validation_cursor: int, episodes: int) -> None:
    if episodes <= 0:
        raise ValueError("validation episodes must be positive")
    if validation_cursor < 0 or validation_cursor + episodes > VALIDATION_TIER_STRIDE:
        raise ValueError(
            "validation window leaves or exhausts its per-tier namespace; "
            "choose a fresh non-overlapping initial cursor"
        )


@torch.no_grad()
def validate(
    policy: UniversalGhostlinePolicy,
    tier: int,
    episodes: int,
    device: torch.device,
    *,
    validation_offset: int = 0,
) -> float:
    was_training = policy.training
    policy.eval()
    successes = 0
    for episode in range(episodes):
        seed = validation_seed(tier, validation_offset + episode)
        env = GhostlineEnv(seed=seed, tier=tier)
        obs, _ = env.reset(seed=seed)
        hidden = None
        terminated = truncated = False
        while not (terminated or truncated):
            action, hidden = policy.act(obs, hidden, deterministic=True, device=device)
            obs, _, terminated, truncated, info = env.step(action)
        successes += int(info["is_success"])
        env.close()
    policy.train(was_training)
    return successes / max(1, episodes)


def validate_curriculum(
    policy: UniversalGhostlinePolicy,
    current_tier: int,
    episodes: int,
    device: torch.device,
    *,
    validation_offset: int = 0,
) -> dict[int, float]:
    return {
        tier: validate(
            policy,
            tier,
            episodes,
            device,
            validation_offset=validation_offset,
        )
        for tier in range(1, current_tier + 1)
    }


def validation_selection_score(rates: dict[int, float]) -> tuple[float, float] | None:
    """Score only complete six-tier gates; partial curricula cannot become champion."""
    if not FULL_VALIDATION_TIERS.issubset(rates):
        return None
    covered_rates = {tier: rates[tier] for tier in FULL_VALIDATION_TIERS}
    return min(covered_rates.values()), covered_rates[6]


def next_acceptance_passes(
    curriculum: AdaptiveCurriculum,
    rates: dict[int, float],
    previous: int,
) -> int:
    """Count only consecutive full-distribution acceptance validations at tier six."""
    if curriculum.current_tier == 6 and curriculum.acceptance_met(rates):
        return previous + 1
    return 0


def _save_training_state(
    path: Path,
    policy: UniversalGhostlinePolicy,
    optimizer: torch.optim.Optimizer,
    rnd: RandomNetworkDistillation,
    rnd_optimizer: torch.optim.Optimizer,
    curriculum: AdaptiveCurriculum,
    *,
    steps: int,
    rate: float,
    args: argparse.Namespace,
    best_score: tuple[float, float],
    acceptance_passes: int,
    validation_cursor: int,
    resume_contract: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rnd": rnd.state_dict(),
            "rnd_optimizer": rnd_optimizer.state_dict(),
            "recurrent": policy.recurrent,
            "recurrent_size": policy.recurrent_size,
            "steps": steps,
            "success_200": rate,
            "current_tier": curriculum.current_tier,
            "consecutive_passes": curriculum.consecutive_passes,
            "validation_history": curriculum.validation_history,
            "best_score": best_score,
            "acceptance_passes": acceptance_passes,
            "validation_cursor": validation_cursor,
            "resume_contract": resume_contract,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "config": vars(args),
            "observation_contract": OBSERVATION_CONTRACT,
            "environment_fingerprint": current_environment_fingerprint(),
        },
        path,
    )


def _load_initial_policy(args: argparse.Namespace, device: torch.device) -> UniversalGhostlinePolicy:
    recurrent = not args.feedforward
    if args.init_checkpoint:
        pretrained = load_policy(args.init_checkpoint, device=device)
        if pretrained.recurrent != recurrent:
            raise ValueError("Initialization checkpoint recurrence does not match --feedforward")
        if pretrained.recurrent_size != args.recurrent_size:
            raise ValueError(
                f"Initialization checkpoint has GRU {pretrained.recurrent_size}, requested {args.recurrent_size}"
            )
        # ``load_policy`` intentionally returns an inference-ready eval model.
        # PPO must switch it back to training mode before cuDNN records the GRU
        # reserve tensors required for recurrent backpropagation.
        return pretrained.train()
    return UniversalGhostlinePolicy(recurrent=recurrent, recurrent_size=args.recurrent_size).to(device)


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    require_validation_window(args.initial_validation_cursor, args.validation_episodes)
    output = Path("artifacts/torchrl") / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / "latest.pt"
    best_path = output / "best.pt"
    rollback_path = output / INITIAL_ROLLBACK_FILENAME
    has_resume = bool(args.resume and resume_path.exists())
    if not has_resume and any(path.exists() for path in (resume_path, best_path, rollback_path)):
        raise RuntimeError(
            "PPO experiment already contains checkpoints; resume it or choose a new experiment name"
        )

    curriculum = AdaptiveCurriculum(current_tier=args.initial_curriculum_tier)
    if not args.curriculum:
        curriculum.current_tier = args.fixed_tier

    def make_vector_env():
        fixed = None if args.curriculum else args.fixed_tier
        return make_curriculum_vector_env(
            env_count=args.envs,
            curriculum=curriculum,
            training_lesson=args.training_lesson,
            fixed_tier=fixed,
            async_envs=args.async_envs,
        )

    policy = _load_initial_policy(args, device)
    rollback_sha256, anchored_initialization = _prepare_initial_rollback(
        policy,
        rollback_path,
        source=args.init_checkpoint,
        allow_existing=has_resume,
    )
    resume_contract = _resume_contract(args, rollback_sha256)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=1e-4)
    rnd = RandomNetworkDistillation().to(device)
    rnd_optimizer = torch.optim.AdamW(rnd.predictor.parameters(), lr=args.rnd_learning_rate, eps=1e-5)
    steps = 0
    restored_best_score: tuple[float, float] = (-1.0, -1.0)
    acceptance_passes = 0
    validation_cursor = args.initial_validation_cursor
    if has_resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        require_current_checkpoint(checkpoint, path=resume_path)
        _require_matching_resume_contract(checkpoint, resume_contract)
        if int(checkpoint.get("recurrent_size", args.recurrent_size)) != args.recurrent_size:
            raise ValueError("Resume checkpoint recurrent size differs from requested configuration")
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "rnd" in checkpoint:
            rnd.load_state_dict(checkpoint["rnd"])
            rnd_optimizer.load_state_dict(checkpoint["rnd_optimizer"])
        steps = int(checkpoint.get("steps", 0))
        curriculum.current_tier = int(checkpoint.get("current_tier", curriculum.current_tier))
        curriculum.consecutive_passes = int(checkpoint.get("consecutive_passes", 0))
        curriculum.validation_history = checkpoint.get("validation_history", curriculum.validation_history)
        restored_best_score = tuple(checkpoint.get("best_score", restored_best_score))
        acceptance_passes = int(checkpoint.get("acceptance_passes", 0))
        if "validation_cursor" not in checkpoint:
            raise RuntimeError("PPO resume state omitted its current validation cursor")
        validation_cursor = int(checkpoint["validation_cursor"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_states = checkpoint.get("torch_cuda_rng_states")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if acceptance_passes < 2:
            require_validation_window(validation_cursor, args.validation_episodes)
    writer = SummaryWriter(output / "tensorboard")
    # Vector workers must be created only after a resumed curriculum tier is
    # restored. Otherwise every child retains the pre-resume tier-one copy.
    envs = make_vector_env()
    obs, _ = envs.reset(seed=[args.seed + i for i in range(args.envs)])
    hidden_size = policy.recurrent_size if policy.recurrent else 384
    hidden = torch.zeros(1, args.envs, hidden_size, device=device) if policy.recurrent else None
    anchor_policy = None
    anchor_hidden = None
    if anchored_initialization and args.anchor_kl_coef > 0.0:
        anchor_policy = load_policy(rollback_path, device=device)
        for parameter in anchor_policy.parameters():
            parameter.requires_grad_(False)
        anchor_hidden = (
            torch.zeros(1, args.envs, anchor_policy.recurrent_size, device=device)
            if anchor_policy.recurrent
            else None
        )
    started, update = time.monotonic(), 0
    starting_steps = steps
    best_score = restored_best_score
    episode_success: list[float] = []
    rate = 0.0
    coefficient = decaying_rnd_coefficient(
        steps,
        initial=args.rnd_coef,
        decay_steps=args.rnd_decay_steps,
        final_fraction=args.rnd_final_fraction,
    )
    acceptance_reached = acceptance_passes >= 2

    def should_continue() -> bool:
        within_time = args.seconds <= 0 or time.monotonic() - started < args.seconds
        within_steps = args.max_steps <= 0 or steps < args.max_steps
        return within_time and within_steps and not acceptance_reached

    while should_continue():
        storage: dict[str, list[torch.Tensor]] = {key: [] for key in OBS_KEYS}
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        extrinsic_rewards: list[torch.Tensor] = []
        intrinsic_rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        initial_hidden = hidden.detach().clone() if hidden is not None else None
        initial_anchor_hidden = (
            anchor_hidden.detach().clone() if anchor_hidden is not None else None
        )
        for _ in range(args.rollout):
            tensors = _tensor_obs(obs, device)
            with torch.no_grad():
                logits, value, next_hidden = policy(tensors, hidden)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                next_anchor_hidden = None
                if anchor_policy is not None:
                    _, _, next_anchor_hidden = anchor_policy(tensors, anchor_hidden)
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            next_tensors = _tensor_obs(next_obs, device)
            coefficient = decaying_rnd_coefficient(
                steps,
                initial=args.rnd_coef,
                decay_steps=args.rnd_decay_steps,
                final_fraction=args.rnd_final_fraction,
            )
            intrinsic = rnd.intrinsic_reward(next_tensors, update_stats=True) if coefficient else torch.zeros(args.envs, device=device)
            intrinsic = mask_terminal_intrinsic_rewards(intrinsic, done)
            extrinsic = torch.as_tensor(reward, dtype=torch.float32, device=device)
            combined = extrinsic + coefficient * intrinsic
            for key in OBS_KEYS:
                storage[key].append(tensors[key].cpu())
            actions.append(action.cpu())
            log_probs.append(distribution.log_prob(action).cpu())
            values.append(value.cpu())
            rewards.append(combined.cpu())
            extrinsic_rewards.append(extrinsic.cpu())
            intrinsic_rewards.append(intrinsic.cpu())
            dones.append(torch.as_tensor(done, dtype=torch.float32))
            hidden = next_hidden.detach() if next_hidden is not None else None
            anchor_hidden = (
                next_anchor_hidden.detach() if next_anchor_hidden is not None else None
            )
            if hidden is not None and done.any():
                hidden[:, torch.as_tensor(done, device=device), :] = 0.0
            if anchor_hidden is not None and done.any():
                anchor_hidden[:, torch.as_tensor(done, device=device), :] = 0.0
            if done.any():
                episode_success.extend(completed_episode_successes(infos, done))
            obs = next_obs
            steps += args.envs

        with torch.no_grad():
            _, bootstrap, _ = policy(_tensor_obs(obs, device), hidden)
        reward_t = torch.stack(rewards).to(device)
        value_t = torch.stack(values).to(device)
        done_t = torch.stack(dones).to(device)
        advantages = torch.zeros_like(reward_t)
        last = torch.zeros(args.envs, device=device)
        for tick in reversed(range(args.rollout)):
            following = bootstrap if tick == args.rollout - 1 else value_t[tick + 1]
            delta = reward_t[tick] + args.gamma * following * (1 - done_t[tick]) - value_t[tick]
            last = delta + args.gamma * args.gae_lambda * (1 - done_t[tick]) * last
            advantages[tick] = last
        returns = advantages + value_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_obs = {key: torch.stack(storage[key]).to(device) for key in OBS_KEYS}
        action_t = torch.stack(actions).to(device)
        old_log_t = torch.stack(log_probs).to(device)
        reset_mask = torch.zeros_like(done_t)
        reset_mask[1:] = done_t[:-1]

        reference_logits = None
        if anchor_policy is not None:
            with torch.no_grad():
                reference_logits, _, _ = anchor_policy.forward_sequence(
                    batch_obs,
                    initial_anchor_hidden,
                    reset_mask=reset_mask,
                )
        anchor_coefficient = decaying_rnd_coefficient(
            steps,
            initial=args.anchor_kl_coef,
            decay_steps=args.anchor_decay_steps,
            final_fraction=args.anchor_final_fraction,
        )

        for _ in range(args.epochs):
            logits, new_value, _ = policy.forward_sequence(batch_obs, initial_hidden, reset_mask=reset_mask)
            dist = torch.distributions.Categorical(logits=logits)
            new_log = dist.log_prob(action_t)
            ratio = (new_log - old_log_t).exp()
            clipped = ratio.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = 0.5 * (new_value - returns).square().mean()
            entropy = dist.entropy().mean()
            anchor_kl = (
                categorical_anchor_kl(logits, reference_logits).mean()
                if reference_logits is not None
                else torch.zeros((), device=device)
            )
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + anchor_coefficient * anchor_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.rnd_coef > 0.0:
                rnd_loss = rnd.predictor_loss(batch_obs, update_fraction=args.rnd_update_fraction)
                rnd_optimizer.zero_grad(set_to_none=True)
                rnd_loss.backward()
                nn.utils.clip_grad_norm_(rnd.predictor.parameters(), args.max_grad_norm)
                rnd_optimizer.step()
            else:
                rnd_loss = torch.zeros((), device=device)

        update += 1
        rate = float(np.mean(episode_success[-200:])) if episode_success else 0.0
        elapsed = max(time.monotonic() - started, 1e-6)
        writer.add_scalar("train/success_200", rate, steps)
        writer.add_scalar("loss/total", float(loss.detach()), steps)
        writer.add_scalar("loss/policy", float(policy_loss.detach()), steps)
        writer.add_scalar("loss/value", float(value_loss.detach()), steps)
        writer.add_scalar("loss/rnd_predictor", float(rnd_loss.detach()), steps)
        writer.add_scalar("loss/anchor_kl", float(anchor_kl.detach()), steps)
        writer.add_scalar("train/entropy", float(entropy.detach()), steps)
        writer.add_scalar("train/anchor_kl_coefficient", anchor_coefficient, steps)
        writer.add_scalar(
            "train/clip_fraction",
            float(((ratio - 1.0).abs() > args.clip_ratio).float().mean().detach()),
            steps,
        )
        writer.add_scalar(
            "train/approx_kl",
            float((old_log_t - new_log).mean().detach()),
            steps,
        )
        writer.add_scalar("reward/extrinsic_mean", float(torch.stack(extrinsic_rewards).mean()), steps)
        writer.add_scalar("reward/intrinsic_mean", float(torch.stack(intrinsic_rewards).mean()), steps)
        writer.add_scalar("reward/rnd_coefficient", coefficient, steps)
        writer.add_scalar("curriculum/tier", curriculum.current_tier, steps)
        writer.add_scalar("train/steps_per_second", (steps - starting_steps) / elapsed, steps)

        validation_rates: dict[int, float] = {}
        if update % args.validation_interval == 0:
            require_validation_window(validation_cursor, args.validation_episodes)
            validation_rates = validate_curriculum(
                policy,
                curriculum.current_tier,
                args.validation_episodes,
                device,
                validation_offset=validation_cursor,
            )
            validation_cursor += args.validation_episodes
            active_rate = validation_rates[curriculum.current_tier]
            promoted = args.curriculum and curriculum.observe_validation(curriculum.current_tier, active_rate)
            for tier, validation_rate in validation_rates.items():
                writer.add_scalar(f"validation/tier_{tier}", validation_rate, steps)
            if promoted:
                envs.close()
                envs = make_vector_env()
                obs, _ = envs.reset()
                if hidden is not None:
                    hidden.zero_()
                if anchor_hidden is not None:
                    anchor_hidden.zero_()
            acceptance_passes = next_acceptance_passes(curriculum, validation_rates, acceptance_passes)
            writer.add_scalar("validation/acceptance_consecutive_passes", acceptance_passes, steps)
            acceptance_reached = acceptance_passes >= 2
        selection = validation_selection_score(validation_rates)
        if selection is not None and selection > best_score:
            best_score = selection
            save_policy(
                policy,
                output / "best.pt",
                steps=steps,
                success_200=rate,
                worst_tier_validation=selection[0],
                tier6_validation=selection[1],
                validation_tiers=sorted(validation_rates),
                selection_stage="complete_six_tier_validation",
                framework="pytorch-2.13-recurrent-ppo",
                rnd_coef=args.rnd_coef,
            )
        if update % args.checkpoint_interval == 0:
            _save_training_state(
                resume_path,
                policy,
                optimizer,
                rnd,
                rnd_optimizer,
                curriculum,
                steps=steps,
                rate=rate,
                args=args,
                best_score=best_score,
                acceptance_passes=acceptance_passes,
                validation_cursor=validation_cursor,
                resume_contract=resume_contract,
            )
            (output / "state.json").write_text(
                json.dumps(
                    {
                        "steps": steps,
                        "success_200": rate,
                        "tier": curriculum.current_tier,
                        "best_worst_tier_validation": best_score[0],
                        "best_tier6_validation": best_score[1],
                        "validation_cursor": validation_cursor,
                        "initial_validation_cursor": args.initial_validation_cursor,
                        "initial_curriculum_tier": args.initial_curriculum_tier,
                        "initial_rollback": str(rollback_path),
                        "initial_rollback_sha256": rollback_sha256,
                        "rnd_coefficient": coefficient,
                        "acceptance_passes": acceptance_passes,
                        "acceptance_reached": acceptance_reached,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    _save_training_state(
        resume_path,
        policy,
        optimizer,
        rnd,
        rnd_optimizer,
        curriculum,
        steps=steps,
        rate=rate,
        args=args,
        best_score=best_score,
        acceptance_passes=acceptance_passes,
        validation_cursor=validation_cursor,
        resume_contract=resume_contract,
    )
    writer.close()
    envs.close()
    return best_path if best_path.exists() else rollback_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ghostline recurrent PPO + RND fine-tuning")
    parser.add_argument("--experiment", default="ghostline-universal")
    parser.add_argument("--seconds", type=int, default=86_400, help="0 disables the wall-clock limit")
    parser.add_argument("--max-steps", type=int, default=0, help="0 disables the environment-step limit")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rnd-coef", type=float, default=0.005)
    parser.add_argument("--rnd-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rnd-decay-steps", type=int, default=5_000_000)
    parser.add_argument("--rnd-final-fraction", type=float, default=0.05)
    parser.add_argument("--rnd-update-fraction", type=float, default=0.25)
    parser.add_argument("--anchor-kl-coef", type=float, default=0.02)
    parser.add_argument("--anchor-decay-steps", type=int, default=2_000_000)
    parser.add_argument("--anchor-final-fraction", type=float, default=0.10)
    parser.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    parser.add_argument("--feedforward", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--async-envs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-tier", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument(
        "--initial-validation-cursor",
        type=int,
        default=0,
        help="first unused per-tier validation offset; persisted and never wrapped",
    )
    parser.add_argument(
        "--initial-curriculum-tier",
        type=int,
        choices=range(1, 7),
        default=1,
        help="fresh-run curriculum tier (use 6 for a qualified universal initializer)",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--training-lesson", type=int, choices=range(0, 8), default=0)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


if __name__ == "__main__":
    print(train(parse_args()))
