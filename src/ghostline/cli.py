from __future__ import annotations

import argparse
from pathlib import Path

from ghostline import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ghostline", description="Ghostline game and reinforcement-learning benchmark")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    play = subparsers.add_parser("play", help="Launch the game")
    play.add_argument("--tier", type=int, choices=range(1, 7))
    play.add_argument("--seed", type=int)
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
        default=7_000_000,
        help="reserved one-shot release slice (7,000,000 by default; 2M-6M are retired)",
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
        default=Path("benchmarks/neural/champion-final-7m-500.json"),
    )
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "menu"
    if command in ("menu", "play", "lab"):
        from ghostline.app import GameApp

        mode = "menu" if command == "menu" else command
        raise SystemExit(GameApp(initial_tier=getattr(args, "tier", None), seed=getattr(args, "seed", None), mode=mode).run())
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
