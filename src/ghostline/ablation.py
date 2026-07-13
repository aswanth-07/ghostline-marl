"""Equal-budget Ghostline policy ablation runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from ghostline.evaluation import evaluate


def _training_command(
    name: str,
    *,
    steps: int,
    envs: int,
    recurrent_size: int,
    init_checkpoint: Path | None,
    feedforward: bool = False,
    rnd: bool = False,
    seed: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "ghostline.torchrl_train",
        f"--experiment=ablation-{name}-seed-{seed}",
        "--seconds=0",
        f"--max-steps={steps}",
        f"--envs={envs}",
        f"--recurrent-size={recurrent_size}",
        f"--seed={seed}",
        "--no-resume",
        f"--rnd-coef={0.005 if rnd else 0.0}",
    ]
    if init_checkpoint is not None:
        command.append(f"--init-checkpoint={init_checkpoint}")
    if feedforward:
        command.append("--feedforward")
    return command


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "ghostline-equal-budget-ablation-v1",
        "environment_steps_per_trainable_condition": args.steps,
        "seeds": args.seeds,
        "conditions": {},
    }
    baseline_conditions = ("random", "teacher")
    for baseline in baseline_conditions:
        destination = output / f"{baseline}.json"
        if args.dry_run:
            report["conditions"][baseline] = {"status": "planned", "output": str(destination)}
        else:
            result = evaluate(
                model=None,
                baseline=baseline,
                episodes=args.evaluation_episodes,
                tier=None,
                output=destination,
                seed_start=2_000_000,
                slice_manifest=None,
            )
            report["conditions"][baseline] = {"status": "complete", "evaluation": result}

    checkpoints = {
        "bc-only": args.bc_checkpoint,
        "bc-dagger-only": args.dagger_checkpoint,
    }
    for name, checkpoint in checkpoints.items():
        if checkpoint is None or not checkpoint.exists():
            report["conditions"][name] = {"status": "missing_checkpoint"}
        elif args.dry_run:
            report["conditions"][name] = {"status": "planned", "checkpoint": str(checkpoint)}
        else:
            result = evaluate(
                model=checkpoint,
                episodes=args.evaluation_episodes,
                tier=None,
                output=output / f"{name}.json",
                seed_start=2_000_000,
                slice_manifest=None,
            )
            report["conditions"][name] = {"status": "complete", "evaluation": result}

    specs = (
        ("pure-ppo-recurrent", None, False, False),
        ("pure-ppo-feedforward", None, True, False),
        ("bc-plus-ppo", args.bc_checkpoint, False, False),
        ("bc-dagger-plus-ppo-rnd", args.dagger_checkpoint, False, True),
    )
    for name, initialization, feedforward, rnd in specs:
        if initialization is not None and not initialization.exists():
            report["conditions"][name] = {"status": "missing_checkpoint"}
            continue
        runs = []
        for seed in range(args.seeds):
            command = _training_command(
                name,
                steps=args.steps,
                envs=args.envs,
                recurrent_size=args.recurrent_size,
                init_checkpoint=initialization,
                feedforward=feedforward,
                rnd=rnd,
                seed=seed,
            )
            run = {"seed": seed, "command": command}
            if not args.dry_run:
                subprocess.run(command, check=True)
                checkpoint = Path("artifacts/torchrl") / f"ablation-{name}-seed-{seed}" / "best.pt"
                result = evaluate(
                    model=checkpoint,
                    episodes=args.evaluation_episodes,
                    tier=None,
                    output=output / f"{name}-seed-{seed}.json",
                    seed_start=2_000_000,
                    slice_manifest=None,
                )
                run.update(checkpoint=str(checkpoint), evaluation=result)
            runs.append(run)
        report["conditions"][name] = {"status": "planned" if args.dry_run else "complete", "runs": runs}
        (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run equal-environment-step Ghostline ablations")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ablations"))
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--recurrent-size", type=int, choices=(256, 384), default=384)
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--dagger-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_ablation(parse_args()), indent=2))
