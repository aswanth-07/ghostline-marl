"""Build-gate smoke test for the base Ghostline wheel in an isolated venv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import venv


ROOT = Path(__file__).resolve().parents[1]


def _venv_python(directory: Path) -> Path:
    if sys.platform == "win32":
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def verify(wheel: Path) -> dict[str, object]:
    wheel = wheel.expanduser().resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"wheel does not exist: {wheel}")
    with tempfile.TemporaryDirectory(prefix="ghostline-clean-install-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _venv_python(environment)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([str(python), "-m", "pip", "check"], cwd=ROOT, check=True)
        probe = textwrap.dedent(
            """
            import json
            import os
            import sys
            from importlib import metadata, util

            import ghostline
            assert "pygame" not in sys.modules
            pygame_was_deferred = "pygame" not in sys.modules
            assert "torch" not in sys.modules
            assert util.find_spec("torch") is None
            assert util.find_spec("onnxruntime") is None
            assert util.find_spec("neon_arena") is None

            import gymnasium as gym
            env = gym.make("GhostlineEnv-v2", tier=1, seed=10101)
            observation, info = env.reset(seed=10101, options={"training_lesson": 1})
            assert observation["action_mask"].shape == (36,)
            env.step(0)
            env.close()
            adaptive = gym.make("GhostlineEnv-v3", tier=6, seed=10102, directive="ghost")
            adaptive_observation, adaptive_info = adaptive.reset(seed=10102)
            assert adaptive_observation["action_mask"].shape == (72,)
            assert adaptive_observation["directive"].shape == (6,)
            assert adaptive_info["contract"] == "GhostlineEnv-v3"
            adaptive.step(0)
            adaptive.close()

            from ghostline.resources import runtime_asset_path
            runtime_assets = (
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "assets/licenses.json",
                "assets/visual/ghostline-environment-atlas-v1.png",
                "assets/visual/ghostline-character-security-atlas-v1.png",
                "assets/visual/ghostline-diagonal-locomotion-v2.png",
            )
            for relative in runtime_assets:
                with runtime_asset_path(relative) as asset:
                    assert asset is not None and asset.is_file(), relative
            with runtime_asset_path("assets/licenses.json") as asset_manifest:
                asset_data = json.loads(asset_manifest.read_text(encoding="utf-8"))
            assert asset_data["project"] == "Ghostline"
            assert asset_data["runtime_distribution"]["license"] == "MIT"
            assert set(asset_data["runtime_distribution"]["files"]) == {
                "assets/visual/ghostline-environment-atlas-v1.png",
                "assets/visual/ghostline-character-security-atlas-v1.png",
                "assets/visual/ghostline-diagonal-locomotion-v2.png",
            }

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            from ghostline.presentation import GhostlineRenderer
            from ghostline.simulation import GhostlineSimulation
            from ghostline.security_controller import AdaptiveSecurityController
            from ghostline.simulation_v3 import GhostlineSimulationV3
            renderer = GhostlineRenderer(GhostlineSimulation(seed=10101, tier=1), visible=False)
            assert not hasattr(renderer, "_key_art")
            assert renderer._environment_atlas is not None
            assert renderer._character_atlas is not None
            assert renderer._diagonal_locomotion_atlas is not None
            frame = renderer.draw(return_array=True)
            assert frame.shape == (360, 640, 3)
            renderer.close()
            adaptive_sim = GhostlineSimulationV3(seed=10102, tier=6, external_security=True)
            security = AdaptiveSecurityController(adaptive_sim)
            assert security.policy is None
            assert security.adapter is None  # PettingZoo is intentionally outside the base wheel.
            security.update(force=True)
            assert security.last_orders
            security.close()

            scripts = {
                item.name: item.value
                for item in metadata.entry_points(group="console_scripts")
                if item.name in {"ghostline", "blackline-heist"}
            }
            assert scripts == {"ghostline": "ghostline.cli:main"}
            print(json.dumps({
                "ghostline_version": ghostline.__version__,
                "environments": ["GhostlineEnv-v2", "GhostlineEnv-v3"],
                "pygame_deferred_until_asset_probe": pygame_was_deferred,
                "torch_available": util.find_spec("torch") is not None,
                "onnxruntime_available": util.find_spec("onnxruntime") is not None,
                "legacy_package_available": util.find_spec("neon_arena") is not None,
                "console_scripts": scripts,
                "runtime_assets": list(runtime_assets),
                "playable_asset_probe": True,
            }))
            """
        )
        completed = subprocess.run(
            [str(python), "-c", probe],
            cwd=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [str(python), "-m", "ghostline", "--help"],
            cwd=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])


def _default_wheel() -> Path:
    candidates = sorted((ROOT / "dist").glob("ghostline-*.whl"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("no Ghostline wheel found under dist/; run `python -m build` first")
    return candidates[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?", help="Wheel to install; defaults to the newest dist/ wheel")
    arguments = parser.parse_args()
    try:
        print(json.dumps(verify(arguments.wheel or _default_wheel()), indent=2))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
