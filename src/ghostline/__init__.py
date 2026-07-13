"""Ghostline game, simulation, and reinforcement-learning environment."""

import importlib
import sys

__all__: list[str] = []
__version__ = "1.0.0"


def register_env() -> None:
    """Register the public Gymnasium environment once."""
    import gymnasium as gym

    if "GhostlineEnv-v1" not in gym.registry:
        gym.register("GhostlineEnv-v1", entry_point="ghostline.env:GhostlineEnvV1")
    if "GhostlineEnv-v2" not in gym.registry:
        gym.register("GhostlineEnv-v2", entry_point="ghostline.env:GhostlineEnv")


if sys.platform != "emscripten":
    GhostlineEnv = getattr(importlib.import_module("ghostline.env"), "GhostlineEnv")
    __all__.append("GhostlineEnv")
    register_env()
