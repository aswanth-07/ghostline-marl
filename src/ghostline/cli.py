from __future__ import annotations

import argparse
from pathlib import Path

from ghostline import __version__


def _add_runner_v2_training_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runner-v2/ppo"),
    )
    parser.add_argument("--resume", action="store_true")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", type=Path)
    initialization.add_argument("--published-v1-init", type=Path)
    parser.add_argument("--allow-scratch", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-envs", type=int, default=2)
    parser.add_argument(
        "--recurrent-size",
        type=int,
        choices=(256, 384, 512),
        default=384,
    )
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
    parser.add_argument("--training-seed-start", type=int, default=0)
    parser.add_argument("--tiers", default="1,2,3,4,5,6")
    parser.add_argument("--directives", default="standard,ghost,speed,greed")
    parser.add_argument(
        "--ghost-directive-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--no-curriculum",
        dest="adaptive_curriculum",
        action="store_false",
    )
    parser.set_defaults(adaptive_curriculum=True)
    parser.add_argument(
        "--initial-curriculum-tier",
        type=int,
        choices=range(1, 7),
        default=1,
    )
    parser.add_argument(
        "--sync-envs",
        dest="async_envs",
        action="store_false",
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
    )
    parser.add_argument("--security-pool-salt", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the complete campaign and write its manifest without starting workers",
    )
    parser.add_argument(
        "--continue-after-acceptance",
        dest="stop_on_acceptance",
        action="store_false",
    )
    parser.set_defaults(stop_on_acceptance=True)
    parser.add_argument("--cpu", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ghostline", description="Ghostline game and reinforcement-learning benchmark")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    play = subparsers.add_parser("play", help="Launch the game")
    play.add_argument("--tier", type=int, choices=range(1, 7))
    play.add_argument("--seed", type=int)
    play.add_argument("--adaptive", action="store_true", help="use developmental Env-v2 coordinated security")
    play.add_argument("--directive", choices=("standard", "ghost", "speed", "greed"), default="standard")
    lab = subparsers.add_parser("lab", help="Launch Agent Lab")
    lab.add_argument("--tier", type=int, choices=range(1, 7), default=6)
    lab.add_argument("--seed", type=int)
    train = subparsers.add_parser("train", help="Launch autonomous recurrent PPO training")
    train.add_argument("--hours", type=float, default=24.0)
    train.add_argument("--experiment", default="ghostline-universal")
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--init-checkpoint", type=Path)
    train.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    train.add_argument("--rnd-coef", type=float, default=0.005)
    train.add_argument("--training-lesson", type=int, choices=range(0, 8), default=0)
    train.add_argument("--initial-validation-cursor", type=int, default=0)
    train.add_argument("--initial-curriculum-tier", type=int, choices=range(1, 7), default=1)
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train_runner_v2 = subparsers.add_parser(
        "train-runner-v2",
        help="Train the developmental 288-action runner against v2 security",
    )
    _add_runner_v2_training_arguments(train_runner_v2)
    train_security = subparsers.add_parser(
        "train-security",
        help="Train the parameter-shared recurrent adaptive-security team",
    )
    train_security.add_argument("--output", type=Path, default=Path("artifacts/security-mappo"))
    train_security.add_argument("--hours", type=float, default=72.0)
    train_security.add_argument("--max-steps", type=int, default=0)
    train_security.add_argument("--envs", type=int, default=8)
    train_security.add_argument("--rollout", type=int, default=256)
    train_security.add_argument("--epochs", type=int, default=4)
    train_security.add_argument("--tiers", default="3,4,5,6")
    train_security.add_argument("--recurrent-size", type=int, choices=(256, 384), default=256)
    train_security.add_argument("--learning-rate", type=float, default=3e-4)
    train_security.add_argument("--gamma", type=float, default=0.999)
    train_security.add_argument("--gae-lambda", type=float, default=0.98)
    train_security.add_argument("--reward-scale", type=float, default=0.05)
    train_security.add_argument("--entropy-coefficient", type=float, default=0.01)
    train_security.add_argument("--device")
    train_security.add_argument("--validation-interval", type=int, default=100_000)
    train_security.add_argument("--validation-episodes", type=int, default=20)
    train_security.add_argument(
        "--training-seed-start",
        type=int,
        default=10_000_000,
    )
    train_security.add_argument(
        "--initial-validation-cursor",
        type=int,
        default=0,
    )
    train_security.add_argument("--bc-warmup-steps", type=int, default=10_000)
    train_security.add_argument("--bc-warmup-epochs", type=int, default=2)
    train_security.add_argument("--bc-warmup-entropy", type=float, default=0.05)
    train_security.add_argument("--uniform-curriculum", action="store_true")
    train_security.add_argument(
        "--scripted-opponent-fraction",
        type=float,
        default=0.0,
        help="fraction of training episodes using the easier scripted runner; validation always uses --runner-model",
    )
    train_security.add_argument("--no-resume", action="store_true")
    train_security.add_argument("--dry-run", action="store_true")
    train_security.add_argument(
        "--runner-model",
        type=Path,
        default=Path("models/ghostline-policy.pt"),
        help="frozen published-v1 opponent checkpoint",
    )
    train_security.add_argument(
        "--runner-pool",
        type=Path,
        action="append",
        default=[],
        help="repeatable published-v1 or v2 runner checkpoint for league training",
    )
    train_security.add_argument(
        "--init-model",
        type=Path,
        help="initialize a fresh optimizer from a compatible security policy checkpoint",
    )
    train_security.add_argument("--scripted-runner", action="store_true")
    co_train = subparsers.add_parser(
        "co-train-v2",
        help="Train runner PPO and security MAPPO concurrently against frozen league snapshots",
    )
    co_train.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v2-cotraining"),
    )
    co_train.add_argument(
        "--published-runner",
        type=Path,
        default=Path("models/ghostline-policy.pt"),
    )
    co_train.add_argument("--hours", type=float, default=24.0)
    co_train.add_argument("--generations", type=int, default=3)
    co_train.add_argument("--runner-envs", type=int, default=16)
    co_train.add_argument("--security-envs", type=int, default=8)
    co_train.add_argument("--runner-rollout", type=int, default=512)
    co_train.add_argument("--security-rollout", type=int, default=192)
    co_train.add_argument("--runner-learning-rate", type=float, default=5.0e-5)
    co_train.add_argument(
        "--runner-entropy-coefficient",
        type=float,
        default=0.003,
    )
    co_train.add_argument(
        "--runner-initial-curriculum-tier",
        type=int,
        choices=range(1, 7),
        default=3,
    )
    co_train.add_argument(
        "--runner-ghost-directive-fraction",
        type=float,
        default=0.50,
    )
    co_train.add_argument("--monitor-seconds", type=float, default=30.0)
    co_train.add_argument("--runner-max-decisions", type=int, default=0)
    co_train.add_argument("--security-max-steps", type=int, default=0)
    co_train.add_argument(
        "--cpu-thread-limit",
        type=int,
        default=1,
        help="per-trainer numerical-library thread cap",
    )
    co_train.add_argument(
        "--cpu-fraction-limit",
        type=float,
        default=0.50,
        help="maximum fraction of logical CPUs inherited by the trainer tree",
    )
    co_train.add_argument(
        "--resume",
        action="store_true",
        help="resume the first incomplete generation from strict trainer checkpoints",
    )
    co_train.add_argument("--dry-run", action="store_true")
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Consume one explicitly reserved final-test slice exactly once",
    )
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--episodes", type=int, default=500)
    evaluate.add_argument("--tier", type=int, choices=range(1, 7))
    evaluate.add_argument("--baseline", choices=("teacher", "random", "legacy_scripted"), default="teacher")
    evaluate.add_argument("--workers", type=int, default=0, help="0 selects a safe automatic process count")
    evaluate.add_argument(
        "--seed-start",
        type=int,
        default=8_000_000,
        help="reserved one-shot release slice (8,000,000 by default; earlier slices are historical)",
    )
    evaluate.add_argument(
        "--slice-manifest",
        type=Path,
        default=Path("benchmarks/final-test-slices.json"),
        help="tracked one-way slice ledger; reused, unknown, or stale slices fail closed",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/neural/champion-final-8m-500.json"),
    )
    evaluate_security = subparsers.add_parser(
        "evaluate-security",
        help="Consume one reserved v2 security final-test slice exactly once",
    )
    evaluate_security.add_argument("--model", type=Path, required=True)
    evaluate_security.add_argument("--tiers", default="3,4,5,6")
    evaluate_security.add_argument("--episodes-per-tier", type=int, default=500)
    evaluate_security.add_argument("--seed-start", type=int, default=14_000_000)
    evaluate_security.add_argument("--device")
    evaluate_security.add_argument("--runner-model", type=Path, default=Path("models/ghostline-policy.pt"))
    evaluate_security.add_argument("--scripted-runner", action="store_true")
    evaluate_security.add_argument(
        "--slice-manifest",
        type=Path,
        default=Path("benchmarks/security/v2-final-test-slices.json"),
        help="tracked one-way v2 security final-test ledger",
    )
    evaluate_security.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/security/final-test.json"),
    )
    evaluate_runner_v2 = subparsers.add_parser(
        "evaluate-runner-v2",
        help="Evaluate a v2 runner on an explicit held-out final-test slice",
    )
    evaluate_runner_v2.add_argument("--model", type=Path, required=True)
    evaluate_runner_v2.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/runner-v2/final-test-20m.json"),
    )
    evaluate_runner_v2.add_argument("--episodes-per-tier", type=int, default=500)
    evaluate_runner_v2.add_argument("--tiers", default="1,2,3,4,5,6")
    evaluate_runner_v2.add_argument("--directives", default="standard")
    evaluate_runner_v2.add_argument("--seed-start", type=int, default=20_000_000)
    evaluate_runner_v2.add_argument("--workers", type=int, default=0)
    evaluate_runner_v2.add_argument(
        "--slice-manifest",
        type=Path,
        default=Path("benchmarks/runner-v2/final-test-slices.json"),
        help="one-way v2 final-test slice ledger",
    )
    evaluate_runner_v2.add_argument(
        "--security-opponent",
        type=Path,
        action="append",
        default=[],
    )
    evaluate_runner_v2.add_argument("--security-pool-salt", type=int, default=0)
    imitate = subparsers.add_parser("imitate", help="Collect fair-teacher data, behavior-clone, or run DAgger")
    imitate_commands = imitate.add_subparsers(dest="imitation_command", required=True)
    collect = imitate_commands.add_parser("collect", help="Collect observation-only teacher labels")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--tiers", default="1,2,3,4,5,6")
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
    clone = imitate_commands.add_parser("bc", help="Train behavior cloning with auxiliary losses")
    clone.add_argument("--dataset", type=Path, action="append", required=True)
    clone.add_argument("--output", type=Path, required=True)
    clone.add_argument("--updates", type=int, default=20_000)
    clone.add_argument("--batch-size", type=int, default=16)
    clone.add_argument("--sequence-length", type=int, default=64)
    clone.add_argument("--burn-in", type=int, default=32)
    clone.add_argument("--validation-fraction", type=float, default=0.10)
    clone.add_argument("--split-seed", type=int, default=0)
    clone.add_argument("--uniform-window-fraction", type=float, default=0.50)
    clone.add_argument("--validation-windows", type=int, default=128)
    clone.add_argument("--learning-rate", type=float, default=2e-4)
    clone.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    clone.add_argument("--device")
    clone.add_argument("--init-checkpoint", type=Path)
    clone.add_argument("--no-resume", action="store_true")
    dagger = imitate_commands.add_parser("dagger", help="Aggregate policy-induced recovery states")
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
        help="optimizer device; defaults to CUDA when available",
    )
    dagger.add_argument("--collection-workers", type=int, default=12)
    dagger.add_argument("--sequence-length", type=int, default=64)
    dagger.add_argument("--burn-in", type=int, default=32)
    dagger.add_argument("--validation-fraction", type=float, default=0.10)
    dagger.add_argument("--split-seed", type=int, default=0)
    dagger.add_argument("--uniform-window-fraction", type=float, default=0.50)
    dagger.add_argument("--validation-windows", type=int, default=128)
    ablate = subparsers.add_parser("ablate", help="Run equal-environment-step policy ablations")
    ablate.add_argument("--output", type=Path, default=Path("artifacts/ablations"))
    ablate.add_argument("--steps", type=int, default=5_000_000)
    ablate.add_argument("--seeds", type=int, default=3)
    ablate.add_argument("--envs", type=int, default=8)
    ablate.add_argument("--evaluation-episodes", type=int, default=100)
    ablate.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    ablate.add_argument("--bc-checkpoint", type=Path)
    ablate.add_argument("--dagger-checkpoint", type=Path)
    ablate.add_argument("--dry-run", action="store_true")
    record = subparsers.add_parser("record", help="Record a policy demonstration")
    record.add_argument("--model", type=Path)
    record.add_argument("--tier", type=int, choices=range(1, 7), default=6)
    record.add_argument("--seed", type=int, default=2_000_000)
    record.add_argument("--output", type=Path, default=Path("videos/ghostline-demo.mp4"))
    package = subparsers.add_parser("package", help="Build the Windows player executable")
    package.add_argument("--model", type=Path, default=Path("models/ghostline-policy.onnx"))
    package.add_argument(
        "--human-only",
        action="store_true",
        help="Build without the ONNX policy (diagnostic; not a portfolio release)",
    )
    package.add_argument("--dry-run", action="store_true")
    export = subparsers.add_parser("export", help="Export and parity-check an ONNX policy")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--output", type=Path, default=Path("models/ghostline-policy.onnx"))
    export.add_argument("--parity-samples", type=int, default=1000)
    export.add_argument(
        "--quantize",
        action="store_true",
        help="produce and parity-gate a sibling dynamic-INT8 candidate",
    )
    export.add_argument(
        "--quantized-output",
        type=Path,
        help="INT8 candidate path (default: <output-stem>.int8.onnx)",
    )
    export.add_argument(
        "--deployment-output",
        type=Path,
        help="copy the accepted INT8 graph here, or verified FP32 when INT8 is rejected",
    )
    export_runner_v2 = subparsers.add_parser(
        "export-runner-v2",
        help="Export a v2 recurrent runner to ONNX and enforce action parity",
    )
    export_runner_v2.add_argument("--model", type=Path, required=True)
    export_runner_v2.add_argument("--output", type=Path, required=True)
    export_runner_v2.add_argument("--parity-samples", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "menu"
    if command in ("menu", "play", "lab"):
        from ghostline.app import GameApp

        mode = "menu" if command == "menu" else command
        raise SystemExit(
            GameApp(
                initial_tier=getattr(args, "tier", None),
                seed=getattr(args, "seed", None),
                mode=mode,
                adaptive=bool(getattr(args, "adaptive", False)),
                directive=getattr(args, "directive", "standard"),
            ).run()
        )
    if command == "train":
        from ghostline.training import launch_training

        raise SystemExit(
            launch_training(
                hours=args.hours,
                experiment=args.experiment,
                max_steps=args.max_steps,
                init_checkpoint=args.init_checkpoint,
                recurrent_size=args.recurrent_size,
                rnd_coef=args.rnd_coef,
                training_lesson=args.training_lesson,
                initial_validation_cursor=args.initial_validation_cursor,
                initial_curriculum_tier=args.initial_curriculum_tier,
                resume=not args.no_resume,
                dry_run=args.dry_run,
            )
        )
    if command == "train-runner-v2":
        from ghostline.runner_train_v2 import train as train_runner_v2

        print(train_runner_v2(args))
        return
    if command == "train-security":
        from ghostline.marl_train import train_security

        print(
            train_security(
                output=args.output,
                hours=args.hours,
                max_steps=args.max_steps,
                env_count=args.envs,
                rollout=args.rollout,
                epochs=args.epochs,
                tiers=args.tiers,
                recurrent_size=args.recurrent_size,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                reward_scale=args.reward_scale,
                entropy_coefficient=args.entropy_coefficient,
                device=args.device,
                validation_interval=args.validation_interval,
                validation_episodes=args.validation_episodes,
                bc_warmup_steps=args.bc_warmup_steps,
                bc_warmup_epochs=args.bc_warmup_epochs,
                bc_warmup_entropy=args.bc_warmup_entropy,
                adaptive_curriculum=not args.uniform_curriculum,
                resume=not args.no_resume,
                dry_run=args.dry_run,
                runner_checkpoint=None if args.scripted_runner else args.runner_model,
                runner_pool=(
                    None
                    if args.scripted_runner
                    else args.runner_pool
                ),
                init_checkpoint=args.init_model,
                scripted_opponent_fraction=args.scripted_opponent_fraction,
                training_seed_start=args.training_seed_start,
                initial_validation_cursor=args.initial_validation_cursor,
            )
        )
        return
    if command == "evaluate":
        from ghostline.evaluation import evaluate

        evaluate(
            model=args.model,
            episodes=args.episodes,
            tier=args.tier,
            output=args.output,
            baseline=args.baseline,
            workers=args.workers,
            seed_start=args.seed_start,
            slice_manifest=args.slice_manifest,
        )
        return
    if command == "evaluate-security":
        from ghostline.marl_train import evaluate_security_checkpoint

        print(
            evaluate_security_checkpoint(
                model=args.model,
                output=args.output,
                tiers=args.tiers,
                episodes_per_tier=args.episodes_per_tier,
                seed_start=args.seed_start,
                device=args.device,
                runner_checkpoint=None if args.scripted_runner else args.runner_model,
                slice_manifest=args.slice_manifest,
            )
        )
        return
    if command == "co-train-v2":
        from ghostline.co_training import CoTrainingConfig, run_co_training

        result = run_co_training(
            CoTrainingConfig(
                output=args.output.resolve(),
                published_runner=args.published_runner.resolve(),
                hours=args.hours,
                generations=args.generations,
                runner_envs=args.runner_envs,
                security_envs=args.security_envs,
                runner_rollout=args.runner_rollout,
                security_rollout=args.security_rollout,
                runner_learning_rate=args.runner_learning_rate,
                runner_entropy_coefficient=args.runner_entropy_coefficient,
                runner_initial_curriculum_tier=args.runner_initial_curriculum_tier,
                runner_ghost_directive_fraction=args.runner_ghost_directive_fraction,
                monitor_seconds=args.monitor_seconds,
                runner_max_decisions=args.runner_max_decisions,
                security_max_steps=args.security_max_steps,
                cpu_thread_limit=args.cpu_thread_limit,
                cpu_fraction_limit=args.cpu_fraction_limit,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        )
        print(result)
        return
    if command == "evaluate-runner-v2":
        from ghostline.evaluation_v2 import evaluate_runner_v2_checkpoint

        print(
            evaluate_runner_v2_checkpoint(
                args.model,
                args.output,
                episodes_per_tier=args.episodes_per_tier,
                tiers=args.tiers,
                directives=args.directives,
                seed_start=args.seed_start,
                workers=args.workers,
                security_checkpoints=args.security_opponent,
                security_pool_salt=args.security_pool_salt,
                slice_manifest=args.slice_manifest,
            )
        )
        return
    if command == "imitate":
        import torch
        from ghostline.imitation import collect_trajectories, run_dagger, train_behavior_clone

        if args.imitation_command == "collect":
            tiers = tuple(int(value) for value in args.tiers.split(","))
            result = collect_trajectories(
                args.output,
                tiers=tiers,
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
            print(result)
        elif args.imitation_command == "bc":
            training_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
                    device=training_device,
                    resume=not args.no_resume,
                    init_checkpoint=args.init_checkpoint,
                    validation_fraction=args.validation_fraction,
                    split_seed=args.split_seed,
                    uniform_window_fraction=args.uniform_window_fraction,
                    validation_windows=args.validation_windows,
                )
            )
        else:
            training_device = args.training_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
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
                    training_device=training_device,
                    collection_workers=args.collection_workers,
                    sequence_length=args.sequence_length,
                    burn_in=args.burn_in,
                    validation_fraction=args.validation_fraction,
                    split_seed=args.split_seed,
                    uniform_window_fraction=args.uniform_window_fraction,
                    validation_windows=args.validation_windows,
                )
            )
        return
    if command == "ablate":
        from ghostline.ablation import run_ablation

        print(run_ablation(args))
        return
    if command == "record":
        from ghostline.recording import record

        record(model=args.model, tier=args.tier, seed=args.seed, output=args.output)
        return
    if command == "package":
        from ghostline.packaging import build_windows

        raise SystemExit(build_windows(model=None if args.human_only else args.model, dry_run=args.dry_run))
    if command == "export":
        from ghostline.exporting import export_policy

        print(
            export_policy(
                args.model,
                args.output,
                parity_samples=args.parity_samples,
                quantize=args.quantize,
                quantized_output=args.quantized_output,
                deployment_output=args.deployment_output,
            )
        )
        return
    if command == "export-runner-v2":
        from ghostline.exporting_v2 import export_runner_v2_checkpoint

        print(
            export_runner_v2_checkpoint(
                args.model,
                args.output,
                parity_samples=args.parity_samples,
            )
        )
        return
