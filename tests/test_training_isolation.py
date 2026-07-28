"""The training path must not be able to reach the presentation layer.

Headless training and human play share one simulation, so a stray import from
a renderer, an audio mixer, or the app shell into a training module would make
training depend on display state. It would also load SDL on a machine that has
no display at all. These checks run each import in a clean subprocess, because
an in-process check would pass merely because another test already imported
Pygame.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

TRAINING_PATH_MODULES = (
    "ghostline.simulation_v2",
    "ghostline.generation_v2",
    "ghostline.env_v2",
    "ghostline.security_env",
    "ghostline.model_v2",
    "ghostline.security_model",
    "ghostline.runner_train_v2",
    "ghostline.marl_train",
)

PRESENTATION_MODULES = (
    "pygame",
    "ghostline.presentation",
    "ghostline.app",
    "ghostline.audio",
)

_PROBE = """
import importlib, json, sys
importlib.import_module({module!r})
print(json.dumps([name for name in {banned!r} if name in sys.modules]))
"""


def _imported_presentation_modules(module: str) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, banned=PRESENTATION_MODULES)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return __import__("json").loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", TRAINING_PATH_MODULES)
def test_training_module_never_imports_presentation(module: str) -> None:
    leaked = _imported_presentation_modules(module)
    assert leaked == [], f"{module} pulled in {leaked}; training must not depend on the renderer"


def test_presentation_still_imports_the_shared_simulation() -> None:
    """The isolation is one-way on purpose.

    Rendering reads simulation state, so this import must keep working. If it
    ever fails, the two sides have been split into separate rules rather than
    separated by layer, which is the outcome the boundary exists to prevent.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ghostline.presentation, ghostline.simulation_v2; print('ok')",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "ok"
