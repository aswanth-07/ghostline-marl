from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _build_module():
    spec = importlib.util.spec_from_file_location("ghostline_web_static_audit", ROOT / "scripts" / "build_web.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_web = _build_module()


def _minimal_bundle(root: Path) -> None:
    for name in (
        "index.html",
        "ghostline.tar.gz",
        "embed-bridge.mjs",
        "ghostline-shell.mjs",
        "matched-runs.mjs",
        "policy-bridge.mjs",
        "policy-manifest.json",
        "ghostline.css",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "favicon.png",
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"release")
    browserfs = root / build_web.BROWSERFS_WEB_PATH
    browserfs.parent.mkdir(parents=True, exist_ok=True)
    browserfs.write_bytes(b"browserfs")
    browserfs_license = root / build_web.BROWSERFS_LICENSE_PATH
    browserfs_license.parent.mkdir(parents=True, exist_ok=True)
    browserfs_license.write_bytes(b"MIT")
    (root / "policy-manifest.json").write_text(
        '{"available": false, "model_url": null}', encoding="utf-8"
    )


def test_launch_shell_uses_flat_facility_graphics_without_retired_key_art() -> None:
    css = (ROOT / "web" / "static" / "ghostline.css").read_text(encoding="utf-8")
    build_source = (ROOT / "scripts" / "build_web.py").read_text(encoding="utf-8")
    assert "ghostline-key-art.webp" not in css
    assert "ghostline-key-art-web.webp" not in build_source
    assert "repeating-linear-gradient" in css


def test_pygbag_pep723_dependencies_use_browser_repository_names() -> None:
    main = (ROOT / "web" / "main.py").read_text(encoding="utf-8")
    dependency_block = main.split("# /// script", 1)[1].split("# ///", 1)[0]
    # Pygbag 0.9.3 resolves literal module/repository keys here; desktop-style
    # version constraints are not parsed and would produce invalid package URLs.
    assert '"numpy"' in dependency_block
    assert '"gymnasium"' in dependency_block
    assert "==" not in dependency_block
    assert main.index("import numpy") < main.index("import gymnasium")


def test_bundle_rejects_redundant_pygbag_apk(tmp_path: Path) -> None:
    _minimal_bundle(tmp_path)
    (tmp_path / "ghostline.apk").write_bytes(b"unused")
    with pytest.raises(RuntimeError, match="unused Pygbag artifact"):
        build_web.bundle_report(tmp_path)


def test_human_bundle_no_longer_requires_retired_key_art(tmp_path: Path) -> None:
    _minimal_bundle(tmp_path)
    report = build_web.bundle_report(tmp_path)
    assert report["model_available"] is False


def test_bundle_requires_project_and_browser_dependency_notices(tmp_path: Path) -> None:
    _minimal_bundle(tmp_path)
    (tmp_path / "THIRD_PARTY_NOTICES.md").unlink()
    with pytest.raises(RuntimeError, match="THIRD_PARTY_NOTICES"):
        build_web.bundle_report(tmp_path)

    _minimal_bundle(tmp_path)
    (tmp_path / build_web.BROWSERFS_LICENSE_PATH).unlink()
    with pytest.raises(RuntimeError, match="LICENSE"):
        build_web.bundle_report(tmp_path)
