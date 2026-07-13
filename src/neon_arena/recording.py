from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pygame
from stable_baselines3 import PPO
from typing import Any

from neon_arena.env import BlacklineHeistEnv


@dataclass(frozen=True)
class RecordingResult:
    output: Path
    frames: int
    fps: int
    policy: str


class EpisodeVideoRecorder:
    def __init__(
        self,
        *,
        output: Path,
        fps: int,
        show: bool,
    ):
        if not show:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.output = output
        self.fps = fps
        self.show = show
        self.frames = 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(output),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        )

    def close(self) -> None:
        self.writer.close()

    def append_episode(
        self,
        *,
        model: Any,
        seed: int,
        steps: int,
        label: str,
        training_steps: int = 0,
        completed_episodes: int = 0,
        deterministic: bool = True,
        hold_frames: int = 30,
        vec_normalize: Any = None,
    ) -> int:
        env = BlacklineHeistEnv(render_mode="human", seed=seed)
        observation, _ = env.reset()
        clip_frames = 0
        
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        is_recurrent = False
        if model is not None:
            try:
                from sb3_contrib import RecurrentPPO
                if isinstance(model, RecurrentPPO):
                    is_recurrent = True
            except ImportError:
                pass
                
        try:
            for _ in range(steps):
                if model is None:
                    action = env.action_space.sample()
                else:
                    if vec_normalize is not None:
                        obs_to_predict = vec_normalize.normalize_obs(observation)
                    else:
                        obs_to_predict = observation
                        
                    if is_recurrent:
                        action, lstm_states = model.predict(
                            obs_to_predict,
                            state=lstm_states,
                            episode_start=episode_starts,
                            deterministic=deterministic,
                        )
                        episode_starts[0] = False
                    else:
                        action, _ = model.predict(obs_to_predict, deterministic=deterministic)
                        
                observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
                
                if vec_normalize is not None:
                    obs_to_predict = vec_normalize.normalize_obs(observation)
                else:
                    obs_to_predict = observation
                    
                env.set_training_stats(
                    {
                        "steps": training_steps,
                        "fps": self.fps,
                        "mean_reward": env.episode_reward,
                        "recording_label": label,
                        "mlp": _model_snapshot(model, obs_to_predict),
                    }
                )
                if not env.render(process_events=self.show, limit_fps=self.show):
                    break
                self.writer.append_data(_screen_frame(env))
                self.frames += 1
                clip_frames += 1
                if terminated or truncated:
                    break

            for _ in range(hold_frames):
                if vec_normalize is not None:
                    obs_to_predict = vec_normalize.normalize_obs(observation)
                else:
                    obs_to_predict = observation
                    
                env.set_training_stats(
                    {
                        "steps": training_steps,
                        "fps": self.fps,
                        "mean_reward": env.episode_reward,
                        "recording_label": f"{label} // EP {completed_episodes}",
                        "mlp": _model_snapshot(model, obs_to_predict),
                    }
                )
                if not env.render(process_events=self.show, limit_fps=self.show):
                    break
                self.writer.append_data(_screen_frame(env))
                self.frames += 1
                clip_frames += 1
        finally:
            env.close()
        return clip_frames


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def record_episode(
    *,
    output: Path,
    model_path: Path | None,
    seed: int,
    steps: int,
    fps: int,
    show: bool,
    hold_frames: int = 60,
) -> RecordingResult:
    model = None
    policy_name = "random"
    vec_normalize = None
    
    if model_path is not None:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        
        model_resolved = _resolve_checkpoint_path(model_path)
        try:
            model = RecurrentPPO.load(str(model_resolved))
        except Exception:
            model = PPO.load(str(model_resolved))
        policy_name = str(model_path)
        
        vec_normalize_path = model_resolved.parent / f"{model_resolved.stem}_vec_normalize.pkl"
        if vec_normalize_path.exists():
            dummy = DummyVecEnv([lambda: BlacklineHeistEnv(seed=seed)])
            vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy)
            vec_normalize.training = False
            vec_normalize.norm_reward = False

    recorder = EpisodeVideoRecorder(output=output, fps=fps, show=show)
    try:
        recorder.append_episode(
            model=model,
            seed=seed,
            steps=steps,
            label="RANDOM BASELINE" if model is None else "CHECKPOINT EVAL",
            deterministic=True,
            hold_frames=hold_frames,
            vec_normalize=vec_normalize,
        )
    finally:
        recorder.close()
    return RecordingResult(output=output, frames=recorder.frames, fps=fps, policy=policy_name)


def _model_snapshot(model: PPO | None, observation: np.ndarray) -> dict[str, object]:
    if model is None:
        return {}
    from neon_arena.training import mlp_snapshot

    return mlp_snapshot(model, observation)


def _screen_frame(env: BlacklineHeistEnv) -> np.ndarray:
    if env.renderer is None:
        raise RuntimeError("Renderer is not initialized.")
    frame = pygame.surfarray.array3d(env.renderer.screen)
    return np.transpose(frame, (1, 0, 2))
