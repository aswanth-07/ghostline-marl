from __future__ import annotations

import hashlib
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


def _minimal_bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    runtime_url = f"./{build_web.PYGBAG_RUNTIME_ROOT.as_posix()}/"
    (root / "index.html").write_text(
        f'<script src="{runtime_url}pythons.js" data-os="snd,gui"></script>'
        f'<script>config = {{cdn: "{runtime_url}"}};</script>',
        encoding="utf-8",
    )
    runtime_payload = b"runtime"
    monkeypatch.setattr(
        build_web,
        "PYGBAG_RUNTIME_PUBLISHED_SHA256",
        {
            relative: hashlib.sha256(runtime_payload).hexdigest()
            for relative in build_web.PYGBAG_RUNTIME_PUBLISHED_SHA256
        },
    )
    monkeypatch.setattr(build_web, "PYGBAG_LICENSE_SHA256", hashlib.sha256(b"license").hexdigest())
    monkeypatch.setattr(build_web, "CPYTHON_LICENSE_SHA256", hashlib.sha256(b"license").hexdigest())
    for relative in build_web.PYGBAG_RUNTIME_PUBLISHED_SHA256:
        destination = root / build_web.PYGBAG_RUNTIME_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(runtime_payload)
    for relative in (build_web.PYGBAG_LICENSE_PATH, build_web.CPYTHON_LICENSE_PATH):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"license")
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
    assert '"gymnasium"' not in dependency_block
    assert "==" not in dependency_block
    runtime = (ROOT / "web" / "runtime.py").read_text(encoding="utf-8")
    assert "_install_browser_gymnasium_shim()" in runtime
    assert 'ModuleType("gymnasium")' in runtime


def test_pygbag_runtime_is_exactly_locked_local_and_terminal_free() -> None:
    template = (ROOT / "web" / "ghostline.tmpl").read_text(encoding="utf-8")
    assert build_web.PYGBAG_RUNTIME_UPSTREAM_SHA256 == {
        Path("pythons.js"): "6da43e3e62c3db933421b99681e8ef99ed9b0ce1589ed8a0c69b88443278e019",
        Path("cpythonrc.py"): "b8a0b8168b58ef7c38c17d4705c9cbe1751fa667a0a6d26ed26f0537134735de",
        Path("empty.ogg"): "884c20d864222b845aa78fb078ec370f4ddaa203cd92ace28440ed7733403b40",
        Path("cpython312/main.js"): "01c4e4dc7145a482ad259d8272ce73d97b58ec2a141bfb57e620347730d159c7",
        Path("cpython312/main.wasm"): "3cfb882de90feeb367325f0c58731932880c8f424fb5a670b98d035ae862b280",
        Path("cpython312/main.data"): "b068df4d59b06b113cfc3c4d6419bdf699d2c2eeb547c9119e2044c98cdc4a59",
    }
    assert 'src="./runtime/pygbag-0.9.3/pythons.js"' in template
    assert 'cdn: "./runtime/pygbag-0.9.3/"' in template
    assert 'data-os="snd,gui"' in template
    assert "vtx,snd,gui" not in template
    assert "pygame-web.github.io" not in template


def test_pygbag_blanker_patch_is_narrow_and_deterministic() -> None:
    original = (
        b'before\n        } catch (x) {\n'
        b'            console.error("FIXME:", x)\n'
        b'        }\nafter'
    )
    patched = build_web._patch_pygbag_bootstrap(original)
    assert b'console.error("FIXME:", x)' not in patched
    assert b"Cross-origin portfolio parents intentionally expose no Pygbag blanker" in patched
    assert patched.startswith(b"before\n") and patched.endswith(b"\nafter")
    with pytest.raises(RuntimeError, match="no longer matches"):
        build_web._patch_pygbag_bootstrap(b"upstream changed")


def test_bundle_rejects_redundant_pygbag_apk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _minimal_bundle(tmp_path, monkeypatch)
    (tmp_path / "ghostline.apk").write_bytes(b"unused")
    with pytest.raises(RuntimeError, match="unused Pygbag artifact"):
        build_web.bundle_report(tmp_path)


def test_human_bundle_no_longer_requires_retired_key_art(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _minimal_bundle(tmp_path, monkeypatch)
    report = build_web.bundle_report(tmp_path)
    assert report["model_available"] is False


def test_bundle_requires_project_and_browser_dependency_notices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _minimal_bundle(tmp_path, monkeypatch)
    (tmp_path / "THIRD_PARTY_NOTICES.md").unlink()
    with pytest.raises(RuntimeError, match="THIRD_PARTY_NOTICES"):
        build_web.bundle_report(tmp_path)

    _minimal_bundle(tmp_path, monkeypatch)
    (tmp_path / build_web.BROWSERFS_LICENSE_PATH).unlink()
    with pytest.raises(RuntimeError, match="LICENSE"):
        build_web.bundle_report(tmp_path)
