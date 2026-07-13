"""Minimal entry point for the Windows player executable.

Keeping this separate from :mod:`ghostline.cli` prevents developer-only
training and recording dependencies from entering the PyInstaller graph.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def bundle_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[2]


def release_smoke_test(*, require_policy: bool = True) -> int:
    """Exercise bundled simulation and ONNX inference without opening a window."""

    from ghostline.env import GhostlineEnv

    env = GhostlineEnv(seed=10101, tier=1)
    try:
        observation, _ = env.reset(seed=10101, options={"training_lesson": 1})
        action = 0
        policy_path = bundle_root() / "models" / "ghostline-policy.onnx"
        if require_policy:
            if not policy_path.is_file():
                raise FileNotFoundError(f"bundled policy is missing: {policy_path}")
            from ghostline.inference import OnnxGhostlinePolicy

            policy = OnnxGhostlinePolicy(policy_path)
            action, _ = policy.act(observation, deterministic=True)
            if not 0 <= action < 36 or not observation["action_mask"][action]:
                raise RuntimeError(f"bundled policy selected illegal action {action}")
        env.step(action)
    finally:
        env.close()
    return 0


def main() -> int:
    if "--release-smoke-test" in sys.argv[1:]:
        return release_smoke_test(require_policy="--human-only" not in sys.argv[1:])

    # App policy/asset lookup uses stable project-relative paths. In a one-file
    # build, PyInstaller extracts those resources beneath ``sys._MEIPASS``.
    os.chdir(bundle_root())
    from ghostline.app import GameApp

    return GameApp(mode="menu").run()


if __name__ == "__main__":
    raise SystemExit(main())
