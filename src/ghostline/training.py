from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _training_args(
    *,
    hours: float,
    experiment: str,
    init_checkpoint: Path | None = None,
    recurrent_size: int = 384,
    rnd_coef: float = 0.005,
    max_steps: int = 0,
    training_lesson: int = 0,
    initial_validation_cursor: int = 0,
    initial_curriculum_tier: int = 1,
    resume: bool = True,
) -> list[str]:
    seconds = 0 if hours <= 0 and max_steps > 0 else max(60, int(hours * 3600))
    args = [
        sys.executable, "-m", "ghostline.torchrl_train",
        f"--experiment={experiment}", f"--seconds={seconds}",
        f"--max-steps={max_steps}", f"--recurrent-size={recurrent_size}",
        f"--rnd-coef={rnd_coef}", f"--training-lesson={training_lesson}",
        f"--initial-validation-cursor={initial_validation_cursor}",
        f"--initial-curriculum-tier={initial_curriculum_tier}",
    ]
    if init_checkpoint is not None:
        args.append(f"--init-checkpoint={init_checkpoint}")
    if not resume:
        args.append("--no-resume")
    return args


def training_command(**kwargs) -> str:
    return subprocess.list2cmdline(_training_args(**kwargs))


def launch_training(*, dry_run: bool = False, **kwargs) -> int:
    command = training_command(**kwargs)
    print(command)
    if dry_run:
        return 0
    return subprocess.run(_training_args(**kwargs), check=False).returncode
