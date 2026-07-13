from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neon_arena.config import MAX_STEPS
from neon_arena.env import BlacklineHeistEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="Launch keyboard play mode.")
    play_parser.add_argument(
        "--curriculum",
        default="full",
        choices=("route", "sequence", "random-city", "sentry", "patrols", "full", "large_easy", "large_sentry_camera", "large_patrols_camera_no_hunters", "large_full", "large_proc_full_no_interact", "large_proc_easy", "large_proc_cameras", "large_proc_patrols", "large_proc_full"),
        help="Curriculum stage to play (controls map size and active entities)."
    )
    play_parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=tuple(range(1, 13)),
        help="Launch campaign stage (1-12). Overrides curriculum stage."
    )

    train_parser = subparsers.add_parser("train", help="Train a Stable-Baselines3 PPO policy.")
    train_parser.add_argument("--timesteps", type=int, default=None)
    train_parser.add_argument("--episodes", type=int, default=None, help="Stop after this many completed episodes.")
    train_parser.add_argument("--n-envs", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--render", action="store_true", help="Render environment 0 during training.")
    train_parser.add_argument("--output", type=Path, default=Path("models/core-runner-v6"))
    train_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    train_parser.add_argument("--resume-from", type=Path, default=None)
    train_parser.add_argument("--no-progress", action="store_true", help="Disable tqdm episode progress.")
    train_parser.add_argument("--curriculum", default="full", help="Curriculum stage to train on, 'auto', or 'auto:stage' to resume curriculum from a specific stage.")
    train_parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate for PPO training.")
    train_parser.add_argument("--ent-coef", type=float, default=0.015, help="Entropy coefficient for PPO training.")
 
    train_record_parser = subparsers.add_parser(
        "train-record",
        help="Train headless and append sparse evaluation clips to one MP4.",
    )
    train_record_parser.add_argument("--timesteps", type=int, default=None)
    train_record_parser.add_argument("--episodes", type=int, default=None, help="Stop after this many completed episodes.")
    train_record_parser.add_argument("--n-envs", type=int, default=16)
    train_record_parser.add_argument("--seed", type=int, default=7)
    train_record_parser.add_argument("--output", type=Path, default=Path("models/core-runner-v6"))
    train_record_parser.add_argument("--video", type=Path, default=Path("videos/training-progress-v6.mp4"))
    train_record_parser.add_argument("--record-every-episodes", type=int, default=100)
    train_record_parser.add_argument("--record-steps", type=int, default=480)
    train_record_parser.add_argument(
        "--record-full-episode",
        action="store_true",
        help=f"Record each evaluation clip until episode end or {MAX_STEPS} env steps.",
    )
    train_record_parser.add_argument("--record-fps", type=int, default=30)
    train_record_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    train_record_parser.add_argument("--resume-from", type=Path, default=None)
    train_record_parser.add_argument("--no-progress", action="store_true", help="Disable tqdm episode progress.")
    train_record_parser.add_argument("--show-recording", action="store_true")
    train_record_parser.add_argument("--deterministic-recording", action="store_true")
    train_record_parser.add_argument("--curriculum", default="full", help="Curriculum stage to train on, 'auto', or 'auto:stage' to resume curriculum from a specific stage.")

    watch_parser = subparsers.add_parser("watch", help="Watch a trained PPO checkpoint.")
    watch_parser.add_argument("--model", type=Path, default=Path("models/core-runner-v6"))
    watch_parser.add_argument("--seed", type=int, default=11)
    watch_parser.add_argument(
        "--stage",
        default="full",
        choices=("route", "sequence", "random-city", "sentry", "patrols", "full", "large_easy", "large_sentry_camera", "large_patrols_camera_no_hunters", "large_full", "large_proc_full_no_interact", "large_proc_easy", "large_proc_cameras", "large_proc_patrols", "large_proc_full"),
        help="Curriculum stage to run model in."
    )

    demo_parser = subparsers.add_parser("random", help="Watch an intentionally weak random policy.")
    demo_parser.add_argument("--seed", type=int, default=13)

    record_parser = subparsers.add_parser("record", help="Record a rendered policy episode to MP4.")
    record_parser.add_argument("--model", type=Path, default=Path("models/core-runner-v6"))
    record_parser.add_argument("--random", action="store_true", help="Record a random-policy episode instead of a model.")
    record_parser.add_argument("--output", type=Path, default=Path("videos/core-runner-recording.mp4"))
    record_parser.add_argument("--seed", type=int, default=17)
    record_parser.add_argument("--steps", type=int, default=MAX_STEPS)
    record_parser.add_argument(
        "--full-episode",
        action="store_true",
        help=f"Record until episode end or {MAX_STEPS} env steps.",
    )
    record_parser.add_argument("--fps", type=int, default=60)
    record_parser.add_argument("--show", action="store_true", help="Show the Pygame window while recording.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint.")
    evaluate_parser.add_argument("--model", type=Path, default=Path("models/core-runner-v6"))
    evaluate_parser.add_argument("--episodes", type=int, default=100)
    evaluate_parser.add_argument("--stage", default="full", choices=("route", "sequence", "random-city", "sentry", "patrols", "full", "large_easy", "large_sentry_camera", "large_patrols_camera_no_hunters", "large_full", "large_proc_full_no_interact", "large_proc_easy", "large_proc_cameras", "large_proc_patrols", "large_proc_full"))
    evaluate_parser.add_argument("--deterministic", action="store_true", default=True)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "play":
        play(args.curriculum, args.stage)
    elif args.command == "train":
        from neon_arena.training import train

        checkpoint = train(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            render=args.render,
            seed=args.seed,
            output=args.output,
            device=args.device,
            target_episodes=args.episodes,
            show_episode_progress=args.episodes is not None and not args.no_progress,
            resume_from=args.resume_from,
            curriculum=args.curriculum,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
        )
        print(f"Saved PPO checkpoint to {checkpoint}.zip")
    elif args.command == "train-record":
        from neon_arena.training import train

        checkpoint = train(
            timesteps=args.timesteps if args.timesteps is not None or args.episodes is not None else 1_000_000,
            n_envs=args.n_envs,
            render=False,
            seed=args.seed,
            output=args.output,
            device=args.device,
            target_episodes=args.episodes,
            show_episode_progress=args.episodes is not None and not args.no_progress,
            resume_from=args.resume_from,
            record_video=args.video,
            record_every_episodes=args.record_every_episodes,
            record_steps=MAX_STEPS if args.record_full_episode else args.record_steps,
            record_fps=args.record_fps,
            show_recording=args.show_recording,
            deterministic_recording=args.deterministic_recording,
            curriculum=args.curriculum,
        )
        print(f"Saved PPO checkpoint to {checkpoint}.zip")
        print(f"Saved training progress video to {args.video}")
    elif args.command == "watch":
        watch(args.model, args.seed, args.stage)
    elif args.command == "random":
        random_policy(args.seed)
    elif args.command == "record":
        record(args)
    elif args.command == "evaluate":
        evaluate(args.model, args.episodes, args.stage, args.deterministic, args.seed)


def play(curriculum: str = "full", stage_num: int | None = None) -> None:
    from neon_arena.progression import load_unlocked_stage, save_unlocked_stage
    
    if stage_num is None and curriculum == "full":
        stage_num = load_unlocked_stage()
        
    if stage_num is not None:
        unlocked = load_unlocked_stage()
        if stage_num > unlocked:
            print(f"Warning: Stage {stage_num} is locked! Highest unlocked stage is Stage {unlocked}.")
            stage_num = unlocked
        env = BlacklineHeistEnv(render_mode="human", action_mode="continuous", campaign_stage=stage_num)
    else:
        env = BlacklineHeistEnv(render_mode="human", action_mode="continuous", curriculum_stage=curriculum)
        
    env.reset()
    env.render()
    try:
        while True:
            action, reset_requested, running = env.renderer.human_action()
            if not running:
                break
            if reset_requested:
                env.reset()
            _, _, terminated, truncated, _ = env.step(action)
            if not env.render(process_events=False):
                break
            if terminated or truncated:
                if env.extracted and getattr(env, "campaign_stage", None) is not None:
                    current_unlocked = load_unlocked_stage()
                    if env.campaign_stage == current_unlocked and current_unlocked < 12:
                        next_stage = current_unlocked + 1
                        save_unlocked_stage(next_stage)
                        print(f"\n* STAGE {env.campaign_stage} CLEARED! UNLOCKED STAGE {next_stage} *")
                        env.campaign_stage = next_stage
                if not terminal_pause(env):
                    break
                env.reset()
    finally:
        env.close()


def random_policy(seed: int) -> None:
    env = BlacklineHeistEnv(render_mode="human", seed=seed)
    env.reset()
    try:
        running = True
        while running:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            running = env.render()
            if terminated or truncated:
                running = terminal_pause(env)
                env.reset()
    finally:
        env.close()


def watch(model_path: Path, seed: int, stage: str = "full") -> None:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_resolved = _resolve_checkpoint_path(model_path)
    try:
        model = RecurrentPPO.load(str(model_resolved))
        is_recurrent = True
    except Exception:
        model = PPO.load(str(model_resolved))
        is_recurrent = False

    env = BlacklineHeistEnv(render_mode="human", seed=seed, curriculum_stage=stage)

    # Validate action space compatibility
    if hasattr(model, "action_space") and model.action_space != env.action_space:
        raise ValueError(
            f"Checkpoint action space mismatch!\n"
            f"Model action space: {model.action_space}\n"
            f"Environment action space: {env.action_space}\n"
            f"Please ensure you are loading the model on the correct stage."
        )
    
    vec_normalize = None
    vec_normalize_path = model_resolved.parent / f"{model_resolved.stem}_vec_normalize.pkl"
    if vec_normalize_path.exists():
        dummy = DummyVecEnv([lambda: BlacklineHeistEnv(seed=seed, curriculum_stage=stage)])
        vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    observation, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    try:
        running = True
        while running:
            if vec_normalize is not None:
                obs_to_predict = vec_normalize.normalize_obs(observation)
            else:
                obs_to_predict = observation

            if is_recurrent:
                action, lstm_states = model.predict(
                    obs_to_predict,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                episode_starts[0] = False
            else:
                action, _ = model.predict(obs_to_predict, deterministic=True)

            observation, _, terminated, truncated, _ = env.step(np.asarray(action))
            running = env.render()
            if terminated or truncated:
                running = terminal_pause(env)
                observation, _ = env.reset()
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
    finally:
        env.close()


def record(args: argparse.Namespace) -> None:
    from neon_arena.recording import record_episode

    model_path = None if args.random else args.model
    if model_path is not None and not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train a model first or use --random for a random-policy recording."
        )
    result = record_episode(
        output=args.output,
        model_path=model_path,
        seed=args.seed,
        steps=MAX_STEPS if args.full_episode else args.steps,
        fps=args.fps,
        show=args.show,
    )
    print(f"Saved {result.frames} frames at {result.fps} FPS to {result.output}")


def terminal_pause(env: BlacklineHeistEnv, frames: int = 90) -> bool:
    for _ in range(frames):
        if not env.render():
            return False
    return True


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def evaluate(model_path: Path, episodes: int, stage: str, deterministic: bool, seed: int) -> None:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from collections import Counter

    model_resolved = _resolve_checkpoint_path(model_path)
    try:
        model = RecurrentPPO.load(str(model_resolved))
        is_recurrent = True
    except Exception:
        model = PPO.load(str(model_resolved))
        is_recurrent = False

    env = BlacklineHeistEnv(render_mode=None, seed=seed, action_mode="discrete", curriculum_stage=stage)
    
    # Validate action space compatibility
    if hasattr(model, "action_space") and model.action_space != env.action_space:
        raise ValueError(
            f"Checkpoint action space mismatch!\n"
            f"Model action space: {model.action_space}\n"
            f"Environment action space: {env.action_space}\n"
            f"Please ensure you are evaluating the model on the correct stage."
        )

    vec_normalize = None
    vec_normalize_path = model_resolved.parent / f"{model_resolved.stem}_vec_normalize.pkl"
    if vec_normalize_path.exists():
        dummy = DummyVecEnv([lambda: BlacklineHeistEnv(seed=seed, curriculum_stage=stage)])
        vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy)
        vec_normalize.training = False
        vec_normalize.norm_reward = False

    successes = 0
    total_reward = 0.0
    total_cores = 0
    total_steps = 0
    total_heat = 0.0
    total_max_heat = 0.0
    fail_reasons: Counter[str] = Counter()

    print(f"Evaluating model '{model_path}' ({'RecurrentPPO' if is_recurrent else 'PPO'}) for {episodes} episodes in stage '{stage}'...")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            if vec_normalize is not None:
                obs_to_predict = vec_normalize.normalize_obs(obs)
            else:
                obs_to_predict = obs

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

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        is_success = info.get("is_success", False)
        cores = info.get("cores", 0)
        fail_reason = info.get("fail_reason", "unknown")
        steps = info.get("episode_step", 0)
        heat = float(info.get("heat", 0.0))
        max_heat = float(info.get("max_heat", 0.0))

        if is_success:
            successes += 1
        else:
            fail_reasons[fail_reason] += 1

        total_reward += ep_reward
        total_cores += cores
        total_steps += steps
        total_heat += heat
        total_max_heat += max_heat

    env.close()

    success_rate = successes / episodes
    avg_reward = total_reward / episodes
    avg_cores = total_cores / episodes
    avg_steps = total_steps / episodes
    avg_heat = total_heat / episodes
    avg_max_heat = total_max_heat / episodes

    print("\n=== Evaluation Results ===")
    print(f"Success Rate:   {success_rate * 100:.2f}% ({successes}/{episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Cores:  {avg_cores:.2f}/3")
    print(f"Average Steps:  {avg_steps:.2f}")
    print(f"Average Heat:   {avg_heat:.1f}")
    print(f"Average MaxHeat:{avg_max_heat:.1f}")
    print("Fail Reasons:")
    if fail_reasons:
        for reason, count in fail_reasons.items():
            print(f"  - {reason}: {count} ({count/episodes*100:.1f}%)")
    else:
        print("  - None")
    print("==========================")
