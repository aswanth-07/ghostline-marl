"""Small setuptools hook for Ghostline's runtime-only asset bundle.

The authored source artwork stays at the repository root for provenance.  A
wheel receives only the alpha-clean runtime derivatives, the asset manifest,
and the MIT license under ``ghostline/_assets``.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent


def _runtime_assets() -> list[tuple[Path, Path]]:
    assets = ROOT / "assets"
    manifest_path = assets / "licenses.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    distribution = manifest["runtime_distribution"]
    if manifest.get("project") != "Ghostline" or distribution.get("license") != "MIT":
        raise ValueError("assets/licenses.json must declare the MIT Ghostline runtime distribution")
    disclosed = {
        details["runtime_file"]
        for details in manifest.get("visual_assets", {}).values()
        if isinstance(details, dict) and isinstance(details.get("runtime_file"), str)
    }
    if disclosed and disclosed != set(distribution["files"]):
        raise ValueError("asset runtime list does not match disclosed runtime_file records")
    selected = [(manifest_path, Path("assets/licenses.json"))]
    assets_root = assets.resolve()
    for value in distribution["files"]:
        source = (ROOT / value).resolve()
        source.relative_to(assets_root)
        if (
            not source.is_file()
            or "-source" in source.name
            or "-web." in source.name
            or "screenshots" in {part.casefold() for part in Path(value).parts}
            or source.name == "ghostline-key-art-menu.png"
        ):
            raise FileNotFoundError(f"invalid declared runtime distribution asset: {source}")
        selected.append((source, Path(value)))
    selected.append((ROOT / "LICENSE", Path("LICENSE")))
    selected.append((ROOT / "THIRD_PARTY_NOTICES.md", Path("THIRD_PARTY_NOTICES.md")))
    selected.append((ROOT / "models" / "ghostline-security.pt", Path("models/ghostline-security.pt")))
    return selected


class BuildPy(_build_py):
    """Copy reviewed runtime assets into the importable wheel package."""

    def run(self) -> None:
        super().run()
        destination = Path(self.build_lib) / "ghostline" / "_assets"
        for source, relative in _runtime_assets():
            if not source.is_file():
                raise FileNotFoundError(f"required Ghostline distribution asset is missing: {source}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


setup(cmdclass={"build_py": BuildPy})
