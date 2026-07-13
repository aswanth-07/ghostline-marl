from __future__ import annotations

import time
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from tqdm.auto import tqdm

from neon_arena.config import MAX_STEPS
from neon_arena.env import BlacklineHeistEnv
from neon_arena.recording import EpisodeVideoRecorder

POLICY_NET_ARCH = (384, 256)


class SpectatorCallback(BaseCallback):
    def __init__(self, *, render: bool, render_every: int = 3):
        super().__init__()
        self.render_enabled = render
        self.render_every = render_every
        self.started_at = time.perf_counter()
        self.recent_returns: deque[float] = deque(maxlen=40)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.recent_returns.append(float(info["episode"]["r"]))
        if not self.render_enabled or self.n_calls % self.render_every != 0:
            return True
        elapsed = max(0.001, time.perf_counter() - self.started_at)
        stats: dict[str, Any] = {
            "steps": self.num_timesteps,
            "fps": self.num_timesteps / elapsed,
            "mean_reward": (
                sum(self.recent_returns) / len(self.recent_returns)
                if self.recent_returns
                else None
            ),
            "mlp": self._mlp_snapshot(),
        }
        self.training_env.env_method("set_training_stats", stats, indices=0)
        return bool(self.training_env.env_method("render", indices=0)[0])

    def _mlp_snapshot(self) -> dict[str, Any]:
        observation = self.locals.get("new_obs")
        if observation is None:
            return {}
        return mlp_snapshot(self.model, np.asarray(observation[0], dtype=np.float32))


class TrainingProgressRecorderCallback(BaseCallback):
    def __init__(
        self,
        *,
        output: Path,
        every_episodes: int,
        clip_steps: int,
        fps: int,
        seed: int,
        show: bool,
        deterministic: bool,
    ):
        super().__init__()
        self.output = output
        self.every_episodes = max(1, every_episodes)
        self.clip_steps = max(1, clip_steps)
        self.fps = fps
        self.seed = seed
        self.show = show
        self.deterministic = deterministic
        self.completed_episodes = 0
        self.next_record_episode = self.every_episodes
        self.recorder: EpisodeVideoRecorder | None = None

    def _on_training_start(self) -> None:
        self.recorder = EpisodeVideoRecorder(output=self.output, fps=self.fps, show=self.show)
        self._record(label="UNTRAINED // EP 0", completed_episodes=0)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.completed_episodes += 1
        while self.completed_episodes >= self.next_record_episode:
            self._record(
                label=f"TRAINING EP {self.next_record_episode}",
                completed_episodes=self.completed_episodes,
            )
            self.next_record_episode += self.every_episodes
        return True

    def _on_training_end(self) -> None:
        self.close()

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def _record(self, *, label: str, completed_episodes: int) -> None:
        if self.recorder is None:
            return
        clip_seed = self.seed + self.next_record_episode + completed_episodes
        self.recorder.append_episode(
            model=self.model,
            seed=clip_seed,
            steps=self.clip_steps,
            label=label,
            training_steps=self.num_timesteps,
            completed_episodes=completed_episodes,
            deterministic=self.deterministic,
            hold_frames=max(10, self.fps // 2),
            vec_normalize=self.training_env,
        )


class EpisodeProgressCallback(BaseCallback):
    def __init__(
        self,
        *,
        target_episodes: int,
        enabled: bool,
        curriculum_mode: str = "full",
        output_path: Path | None = None,
        resume_from: Path | None = None,
    ):
        super().__init__()
        self.target_episodes = max(1, target_episodes)
        self.enabled = enabled
        self.curriculum_mode = curriculum_mode
        self.curriculum_enabled = curriculum_mode.startswith("auto")
        self.output_path = output_path
        
        self.stages = [
            "route",
            "sequence",
            "random-city",
            "sentry",
            "patrols",
            "full",
            "large_easy",
            "large_sentry_camera",
            "large_patrols_camera_no_hunters",
            "large_full",
            "large_proc_easy",
            "large_proc_cameras",
            "large_proc_patrols",
            "large_proc_full"
        ]
        self.stage_thresholds = {
            "route": 0.95,
            "sequence": 0.90,
            "random-city": 0.85,
            "sentry": 0.80,
            "patrols": 0.70,
            "full": 0.70,
            "large_easy": 0.80,
            "large_sentry_camera": 0.75,
            "large_patrols_camera_no_hunters": 0.75,
            "large_full": 0.80,
            "large_proc_easy": 0.80,
            "large_proc_cameras": 0.75,
            "large_proc_patrols": 0.70,
            "large_proc_full": 0.75,
        }
        
        if self.curriculum_enabled:
            extracted_stage = None
            if resume_from is not None:
                filename = resume_from.name
                for stage in self.stages:
                    if f"_stage_{stage}" in filename:
                        extracted_stage = stage
                        break
            
            if ":" in curriculum_mode:
                start_stage = curriculum_mode.split(":")[1]
                if start_stage in self.stages:
                    extracted_stage = start_stage
                    
            if extracted_stage is not None:
                self.current_stage = extracted_stage
                self.current_stage_idx = self.stages.index(extracted_stage)
            else:
                self.current_stage = "route"
                self.current_stage_idx = 0
        else:
            self.current_stage = curriculum_mode
            self.current_stage_idx = self.stages.index(curriculum_mode) if curriculum_mode in self.stages else 5
            
        self.completed_episodes = 0
        self.stage_episodes = 0
        self.latest_return: float | None = None
        self.latest_length: int | None = None
        self.recent_returns: deque[float] = deque(maxlen=50)
        self.recent_successes: deque[bool] = deque(maxlen=100)
        self.recent_cores: deque[int] = deque(maxlen=100)
        self.recent_fail_reasons: deque[str] = deque(maxlen=100)
        self.progress: tqdm | None = None

    def _on_training_start(self) -> None:
        if self.curriculum_enabled:
            self.training_env.env_method("set_curriculum_stage", self.current_stage)
            
        if self.enabled:
            self.progress = tqdm(
                total=self.target_episodes,
                desc="Episodes",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )

    def _evaluate_current_stage(self) -> float:
        # Save imports locally to avoid circular dependencies
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        
        eval_episodes = 100
        # Use separate seed to avoid memorizing layouts
        eval_seed = 9999 + self.current_stage_idx * 1000
        
        eval_env = BlacklineHeistEnv(
            render_mode=None, 
            seed=eval_seed, 
            action_mode="discrete", 
            curriculum_stage=self.current_stage
        )
        
        eval_vec_env = DummyVecEnv([lambda: eval_env])
        
        # Sync VecNormalize stats if training env is normalized
        training_env = self.training_env
        if isinstance(training_env, VecNormalize):
            eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_reward=False)
            eval_vec_env.obs_rms = training_env.obs_rms.copy()
            eval_vec_env.ret_rms = training_env.ret_rms.copy()
            
        is_recurrent = hasattr(self.model, "policy") and hasattr(self.model.policy, "lstm_actor")
        
        successes = 0
        for ep in range(eval_episodes):
            obs = eval_vec_env.reset()
            done = False
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            while not done:
                if is_recurrent:
                    action, lstm_states = self.model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    episode_starts[0] = False
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                
                obs, reward, dones, infos = eval_vec_env.step(action)
                done = dones[0]
                if done:
                    info = infos[0]
                    if info.get("is_success", False):
                        successes += 1
                        
        eval_vec_env.close()
        return successes / eval_episodes

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        episode_infos = [
            info["episode"]
            for info in infos
            if "episode" in info
        ]
        for episode_info in episode_infos:
            self.latest_return = float(episode_info["r"])
            self.latest_length = int(episode_info["l"])
            self.recent_returns.append(self.latest_return)
            
        new_episodes = len(episode_infos)
        if new_episodes <= 0:
            return True
            
        previous_completed = self.completed_episodes
        previous_stage_episodes = self.stage_episodes
        for info in infos:
            if "episode" in info:
                self.completed_episodes += 1
                self.stage_episodes += 1
                
                is_success = info.get("is_success", False)
                cores = info.get("cores", 0)
                fail_reason = info.get("fail_reason", "unknown")
                
                self.recent_successes.append(is_success)
                self.recent_cores.append(cores)
                self.recent_fail_reasons.append(fail_reason)
                
                # Record detailed metrics to TensorBoard logger for this ended episode
                if hasattr(self, "model") and self.model is not None:
                    # Log raw reward components
                    reward_components = [
                        "extraction", "core", "progress", "room_new", "room_objective", 
                        "wrong_room", "time", "heat", "cone", "lock", "hit", "projectile", 
                        "emp", "loot", "timeout", "stall", "proximity"
                    ]
                    for key in reward_components:
                        info_key = f"reward_{key}"
                        if info_key in info:
                            self.logger.record(f"reward/{key}", info[info_key])
                            
                    # Log behavior statistics
                    behavior_episode = ["success", "cores_collected", "extracted", "timed_out", "shield_depleted", "final_heat", "max_heat", "episode_length", "final_room", "timeout_with_3_cores", "missing_cores_on_timeout"]
                    for key in behavior_episode:
                        info_key = f"episode_{key}"
                        if info_key in info:
                            self.logger.record(f"episode/{key}", info[info_key])
                            
                    behavior_route = ["wrong_room_steps", "objective_room_entries", "objective_room_reached", "room_transitions", "rooms_visited_unique", "same_room_max_steps", "next_doorway_reached"]
                    for key in behavior_route:
                        info_key = f"route_{key}"
                        if info_key in info:
                            self.logger.record(f"route/{key}", info[info_key])
                            
                    behavior_combat = ["drone_contacts", "projectile_hits"]
                    for key in behavior_combat:
                        info_key = f"combat_{key}"
                        if info_key in info:
                            self.logger.record(f"combat/{key}", info[info_key])
                            
                    behavior_stealth = ["camera_alerts", "cone_steps"]
                    for key in behavior_stealth:
                        info_key = f"stealth_{key}"
                        if info_key in info:
                            self.logger.record(f"stealth/{key}", info[info_key])
                            
                    behavior_emp = ["uses", "invalid_presses", "wasted_uses", "effective_uses"]
                    for key in behavior_emp:
                        info_key = f"emp_{key}"
                        if info_key in info:
                            self.logger.record(f"emp/{key}", info[info_key])
                            
                    behavior_loot = ["items_collected", "reward_capped"]
                    for key in behavior_loot:
                        info_key = f"loot_{key}"
                        if info_key in info:
                            self.logger.record(f"loot/{key}", info[info_key])
                            
                    behavior_map = ["route_directness", "loop_count", "locked_edge_count", "camera_coverage", "hazard_rooms"]
                    for key in behavior_map:
                        info_key = f"map_{key}"
                        if info_key in info:
                            self.logger.record(f"map/{key}", info[info_key])
                
        # Record curriculum metrics to TensorBoard logger
        if hasattr(self, "model") and self.model is not None:
            self.logger.record("curriculum/stage_idx", self.current_stage_idx)
            if len(self.recent_successes) > 0:
                success_rate = sum(self.recent_successes) / len(self.recent_successes)
                self.logger.record("curriculum/success_rate", success_rate)
            if len(self.recent_cores) > 0:
                avg_cores = sum(self.recent_cores) / len(self.recent_cores)
                self.logger.record("curriculum/average_cores", avg_cores)

        # Handle curriculum promotion using deterministic evaluation
        if self.curriculum_enabled and self.current_stage_idx < len(self.stages) - 1:
            if self.stage_episodes > 0 and (self.stage_episodes // 300 > previous_stage_episodes // 300):
                if self.progress is not None:
                    self.progress.write(f"[Curriculum] Evaluating stage '{self.current_stage}' (episode {self.completed_episodes})...")
                
                eval_success_rate = self._evaluate_current_stage()
                target_threshold = self.stage_thresholds.get(self.current_stage, 0.80)
                
                if self.progress is not None:
                    self.progress.write(f"[Curriculum] Stage '{self.current_stage}' Eval Success: {eval_success_rate * 100:.1f}% (Target: {target_threshold * 100:.1f}%)")
                
                self.logger.record(f"curriculum/eval_success_{self.current_stage}", eval_success_rate)
                
                if eval_success_rate >= target_threshold:
                    old_stage_name = self.current_stage
                    self.current_stage_idx += 1
                    self.current_stage = self.stages[self.current_stage_idx]
                    self.stage_episodes = 0
                    self.recent_successes.clear()
                    self.recent_cores.clear()
                    self.recent_fail_reasons.clear()
                    
                    self.training_env.env_method("set_curriculum_stage", self.current_stage)
                    
                    # Save checkpoint on promotion
                    if self.output_path is not None:
                        checkpoint_name = self.output_path.parent / f"{self.output_path.stem}_stage_{old_stage_name}"
                        self.model.save(str(checkpoint_name))
                        if isinstance(self.training_env, VecNormalize):
                            self.training_env.save(str(checkpoint_name.parent / f"{checkpoint_name.name}_vec_normalize.pkl"))
                        if self.progress is not None:
                            self.progress.write(f"[Curriculum] Saved promoted stage checkpoint to {checkpoint_name}.zip")
                            self.progress.write(f"[Curriculum] Promoted to stage '{self.current_stage}' at episode {self.completed_episodes}")

        # Periodic checkpoint save every 1000 completed episodes
        if new_episodes > 0 and self.completed_episodes > 0 and (self.completed_episodes // 1000 > previous_completed // 1000):
            if self.output_path is not None:
                checkpoint_name = self.output_path.parent / f"{self.output_path.stem}_ep_{self.completed_episodes}"
                if self.progress is not None:
                    self.progress.write(f"[Checkpoint] Saving periodic checkpoint to {checkpoint_name}.zip")
                self.model.save(str(checkpoint_name))
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(str(checkpoint_name.parent / f"{checkpoint_name.name}_vec_normalize.pkl"))

        accepted = max(
            0,
            min(self.completed_episodes, self.target_episodes)
            - min(previous_completed, self.target_episodes),
        )
        
        if self.progress is not None and accepted > 0:
            self.progress.update(accepted)
            self.progress.set_postfix(self._postfix())
            
        return self.completed_episodes < self.target_episodes

    def _on_training_end(self) -> None:
        self.close()

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()
            self.progress = None

    def _postfix(self) -> dict[str, int | str]:
        values: dict[str, int | str] = {
            "steps": self.num_timesteps,
            "completed": min(self.completed_episodes, self.target_episodes),
            "stage": self.current_stage,
        }
        if self.recent_successes:
            success_rate = sum(self.recent_successes) / len(self.recent_successes)
            values["success_100"] = f"{success_rate * 100:.1f}%"
        else:
            values["success_100"] = "0.0%"
            
        if self.recent_cores:
            avg_cores = sum(self.recent_cores) / len(self.recent_cores)
            values["cores_100"] = f"{avg_cores:.2f}"
        else:
            values["cores_100"] = "0.00"
            
        if self.recent_fail_reasons:
            fails = [r for r in self.recent_fail_reasons if r != "none"]
            if fails:
                most_common = max(set(fails), key=fails.count)
                values["fail_reason"] = most_common
            else:
                values["fail_reason"] = "none"
        else:
            values["fail_reason"] = "none"
            
        if self.latest_return is not None:
            mean_return = sum(self.recent_returns) / len(self.recent_returns)
            values["final_reward"] = f"{self.latest_return:.2f}"
            values["mean_reward_50"] = f"{mean_return:.2f}"
        if self.latest_length is not None:
            values["last_len"] = self.latest_length
        return values


def train(
    *,
    timesteps: int | None,
    n_envs: int,
    render: bool,
    seed: int,
    output: Path,
    device: str = "cpu",
    target_episodes: int | None = None,
    show_episode_progress: bool = False,
    resume_from: Path | None = None,
    record_video: Path | None = None,
    record_every_episodes: int = 100,
    record_steps: int = 480,
    record_fps: int = 30,
    record_seed: int | None = None,
    show_recording: bool = False,
    deterministic_recording: bool = False,
    curriculum: str = "full",
    learning_rate: float = 2.5e-4,
    ent_coef: float = 0.015,
) -> Path:
    initial_stage = "route" if curriculum == "auto" else curriculum
    factories = [
        _factory(
            seed=seed + index,
            render_mode="human" if render and index == 0 else None,
            curriculum_stage=initial_stage,
            action_mode="discrete"
        )
        for index in range(n_envs)
    ]
    if render or n_envs <= 1:
        vector_env = DummyVecEnv(factories)
    else:
        vector_env = SubprocVecEnv(factories)
    
    # Set up VecNormalize
    vec_normalize_path = None
    if resume_from is not None:
        model_resolved = _resolve_checkpoint_path(resume_from)
        vec_normalize_path = model_resolved.parent / f"{model_resolved.stem}_vec_normalize.pkl"
        
    if vec_normalize_path is not None and vec_normalize_path.exists():
        vector_env = VecNormalize.load(str(vec_normalize_path), vector_env)
        vector_env.training = True
    else:
        vector_env = VecNormalize(vector_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
    rollout_steps = 512
    tensorboard_log = str(output.parent.parent / "runs")
    model_verbose = 0 if show_episode_progress else 1
    if resume_from is not None:
        model_resolved = _resolve_checkpoint_path(resume_from)
        try:
            model = RecurrentPPO.load(
                str(model_resolved),
                env=vector_env,
                device=device,
                tensorboard_log=tensorboard_log,
                verbose=model_verbose,
            )
        except Exception:
            model = PPO.load(
                str(model_resolved),
                env=vector_env,
                device=device,
                tensorboard_log=tensorboard_log,
                verbose=model_verbose,
            )
        from stable_baselines3.common.utils import get_schedule_fn
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        model.ent_coef = ent_coef
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vector_env,
            seed=seed,
            verbose=model_verbose,
            learning_rate=learning_rate,
            n_steps=rollout_steps,
            batch_size=512 if rollout_steps * n_envs >= 512 else rollout_steps * n_envs,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.97,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            device=device,
            policy_kwargs={
                "net_arch": dict(pi=[384, 256], vf=[384, 256]),
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
                "activation_fn": nn.Tanh,
            },
            tensorboard_log=tensorboard_log,
        )
    callbacks: list[BaseCallback] = [SpectatorCallback(render=render)]
    episode_progress: EpisodeProgressCallback | None = None
    if target_episodes is not None:
        episode_progress = EpisodeProgressCallback(
            target_episodes=target_episodes,
            enabled=show_episode_progress,
            curriculum_mode=curriculum,
            output_path=output,
            resume_from=resume_from,
        )
        callbacks.append(episode_progress)
    progress_recorder: TrainingProgressRecorderCallback | None = None
    if record_video is not None:
        progress_recorder = TrainingProgressRecorderCallback(
            output=record_video,
            every_episodes=record_every_episodes,
            clip_steps=record_steps,
            fps=record_fps,
            seed=seed + 10_000 if record_seed is None else record_seed,
            show=show_recording,
            deterministic=deterministic_recording,
        )
        callbacks.append(progress_recorder)
    callback = CallbackList(callbacks)
    effective_timesteps = timesteps
    if effective_timesteps is None:
        effective_timesteps = target_episodes * MAX_STEPS if target_episodes is not None else 200_000
    try:
        model.learn(
            total_timesteps=effective_timesteps,
            callback=callback,
            reset_num_timesteps=resume_from is None,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        if isinstance(vector_env, VecNormalize):
            vector_env.save(str(output.parent / f"{output.stem}_vec_normalize.pkl"))
    finally:
        if episode_progress is not None:
            episode_progress.close()
        if progress_recorder is not None:
            progress_recorder.close()
        vector_env.close()
    return output


def _factory(*, seed: int, render_mode: str | None, curriculum_stage: str, action_mode: str):
    def create() -> Monitor:
        return Monitor(BlacklineHeistEnv(render_mode=render_mode, seed=seed, action_mode=action_mode, curriculum_stage=curriculum_stage))

    return create


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def mlp_parameter_count() -> int:
    input_size = 256  # Output size of LSTM layer
    action_size = 13

    def dense_count(inputs: int, outputs: int) -> int:
        return inputs * outputs + outputs

    hidden = list(POLICY_NET_ARCH)
    actor_count = dense_count(input_size, hidden[0])
    actor_count += dense_count(hidden[0], hidden[1])
    actor_count += dense_count(hidden[-1], action_size)

    critic_count = dense_count(input_size, hidden[0])
    critic_count += dense_count(hidden[0], hidden[1])
    critic_count += dense_count(hidden[-1], 1)
    return actor_count + critic_count


@torch.no_grad()
def mlp_snapshot(model: Any, observation: np.ndarray) -> dict[str, Any]:
    policy = model.policy
    obs_tensor = torch.as_tensor(observation.reshape(1, -1), device=policy.device)
    features = policy.extract_features(obs_tensor)

    if hasattr(policy, "lstm_actor"):
        p_lstm_out, _ = policy.lstm_actor(features.unsqueeze(0), None)
        policy_input = p_lstm_out.squeeze(0)
        
        if hasattr(policy, "lstm_critic"):
            v_lstm_out, _ = policy.lstm_critic(features.unsqueeze(0), None)
            value_input = v_lstm_out.squeeze(0)
        else:
            value_input = policy_input
    else:
        policy_input = features
        value_input = features

    policy_latent, policy_activations = _forward_with_activations(policy.mlp_extractor.policy_net, policy_input)
    value_latent, value_activations = _forward_with_activations(policy.mlp_extractor.value_net, value_input)
    
    action_out = policy.action_net(policy_latent)
    if hasattr(policy.action_dist, "action_dims"):
        logits = action_out.detach().cpu().numpy()[0]
        splits = np.split(
            logits,
            np.cumsum(policy.action_dist.action_dims)[:-1],
        )
        move_logits = splits[0]
        dash_logits = splits[1]
        emp_logits = splits[2] if len(splits) > 2 else np.array([])
        interact_logits = splits[3] if len(splits) > 3 else np.array([])
        
        move_idx = int(np.argmax(move_logits))
        dash_idx = int(np.argmax(dash_logits))
        emp_idx = int(np.argmax(emp_logits)) if emp_logits.size > 0 else 0
        interact_idx = int(np.argmax(interact_logits)) if interact_logits.size > 0 else 0
    else:
        action_means = action_out.detach().cpu().numpy()[0]
        move_idx = int(np.clip(action_means[0], 0, 8))
        dash_idx = int(np.clip(action_means[1], 0, 1))
        emp_idx = int(np.clip(action_means[2], 0, 1)) if action_means.size > 2 else 0
        interact_idx = int(np.clip(action_means[3], 0, 1)) if action_means.size > 3 else 0

    SQRT2_INV = 0.70710678
    move_mapping = [
        (0.0, 0.0),
        (0.0, -1.0),
        (SQRT2_INV, -SQRT2_INV),
        (1.0, 0.0),
        (SQRT2_INV, SQRT2_INV),
        (0.0, 1.0),
        (-SQRT2_INV, SQRT2_INV),
        (-1.0, 0.0),
        (-SQRT2_INV, -SQRT2_INV)
    ]
    mx, my = move_mapping[np.clip(move_idx, 0, 8)]
    dash_val = 1.0 if dash_idx == 1 else -1.0
    emp_val = 1.0 if emp_idx == 1 else -1.0
    interact_val = 1.0 if interact_idx == 1 else -1.0
    
    value = float(policy.value_net(value_latent).detach().cpu().numpy()[0, 0])

    param_count = sum(p.numel() for p in policy.parameters())

    actions_list = [mx, my, dash_val, emp_val]
    if (hasattr(policy.action_dist, "action_dims") and len(policy.action_dist.action_dims) > 3) or (not hasattr(policy.action_dist, "action_dims") and action_means.size > 3):
        actions_list.append(interact_val)

    return {
        "arch": "84 -> 256 LSTM -> actor/critic heads" if hasattr(policy, "lstm_actor") else f"84 -> {POLICY_NET_ARCH[0]} -> {POLICY_NET_ARCH[1]} -> 13",
        "params": param_count,
        "obs": _sample_values(observation, limit=10),
        "policy_layers": [_sample_values(layer, limit=10) for layer in policy_activations],
        "value_layers": [_sample_values(layer, limit=8) for layer in value_activations],
        "actions": actions_list,
        "value": value,
    }


def _forward_with_activations(network: nn.Sequential, inputs: torch.Tensor) -> tuple[torch.Tensor, list[np.ndarray]]:
    activations: list[np.ndarray] = []
    values = inputs
    for layer in network:
        values = layer(values)
        if isinstance(layer, (nn.Tanh, nn.ReLU, nn.ELU, nn.SiLU)):
            activations.append(values.detach().cpu().numpy()[0])
    return values, activations


def _sample_values(values: np.ndarray, *, limit: int) -> list[float]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size <= limit:
        sampled = array
    else:
        indices = np.linspace(0, array.size - 1, limit, dtype=np.int32)
        sampled = array[indices]
    return [float(np.clip(value, -1.0, 1.0)) for value in sampled]
