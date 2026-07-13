from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
from typing import Iterable
from urllib.request import Request, urlopen
import zlib

from ghostline.onnx_contract import (
    POLICY_INPUT_SHAPES,
    environment_fingerprint,
    validate_onnx_policy,
)


ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = ROOT / ".web-build"
STAGE = BUILD_ROOT / "ghostline"
OUTPUT = STAGE / "build" / "web"
CACHE = BUILD_ROOT / "cache"

PYGBAG_VERSION = "0.9.3"
ORT_VERSION = "1.27.0"
ORT_TARBALL = f"https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{ORT_VERSION}.tgz"
ORT_SHA512 = "ogDLsqIozHZwifPuN37OproAo0byX6t43/bP8GzeZWBWD6MOGExswFAx3up4NS/vvWBOg2u2PXomDt3rMmdQSg=="
ORT_FILES = (
    "dist/ort.all.min.mjs",
    "dist/ort-wasm-simd-threaded.mjs",
    "dist/ort-wasm-simd-threaded.wasm",
    "dist/ort-wasm-simd-threaded.jsep.mjs",
    "dist/ort-wasm-simd-threaded.jsep.wasm",
)
ORT_LICENSE_URL = f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{ORT_VERSION}/LICENSE"
ORT_LICENSE_SHA512 = "KAPoWEawT7C4c8fzJN9DETrXpoK/WIJNb3TUMdsiQvN16YbG/hPJamGWS7NyTQ2HRBpFLjAK+Zxl+JUajfEOEQ=="
ORT_NOTICES_URL = (
    f"https://raw.githubusercontent.com/microsoft/onnxruntime/v{ORT_VERSION}/ThirdPartyNotices.txt"
)
ORT_NOTICES_SHA512 = "qGkIVn51iET5V9b4sCTRf6aMHL8b11tcwkEn24puzrSJud6CVCGOy3JYuNB2fnITg1+Hjc1ytw1jLywK52odSQ=="
BROWSERFS_VERSION = "1.4.3"
BROWSERFS_TARBALL = f"https://registry.npmjs.org/browserfs/-/browserfs-{BROWSERFS_VERSION}.tgz"
BROWSERFS_SHA512 = "tz8HClVrzTJshcyIu8frE15cjqjcBIu15Bezxsvl/i+6f59iNCN3kznlWjz0FEb3DlnDx3gW5szxeT6D1x0s0w=="
BROWSERFS_WEB_PATH = Path("vendor") / f"browserfs-{BROWSERFS_VERSION}.min.js"
BROWSERFS_LICENSE_PATH = Path("licenses") / f"browserfs-{BROWSERFS_VERSION}" / "LICENSE"
ORT_LICENSE_PATH = Path("licenses") / f"onnxruntime-web-{ORT_VERSION}" / "LICENSE"
ORT_NOTICES_PATH = Path("licenses") / f"onnxruntime-web-{ORT_VERSION}" / "ThirdPartyNotices.txt"

DEFAULT_INITIAL_BUDGET = 25 * 1024 * 1024
DEFAULT_WASM_AGENT_BUDGET = 35 * 1024 * 1024
ORT_WEB_ROOT = Path("vendor") / f"onnxruntime-web-{ORT_VERSION}"
MODEL_DIRECTORY = Path("models")
WEB_RUNTIME_MODULES = (
    "__init__.py",
    "app.py",
    "audio.py",
    "config.py",
    "env.py",
    "generation.py",
    "policies.py",
    "presentation.py",
    "progression.py",
    "resources.py",
    "simulation.py",
    "types.py",
)
# Retained as a public audit/compatibility contract for tools that inspect the
# old recursive-copy filter. Production staging is stricter: it copies only
# manifest-declared runtime assets through ``_runtime_asset_paths`` below.
WEB_ASSET_IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    "*-source.*",
    "*-source-*",
    "*-web.*",
    "ghostline-key-art-menu.png",
    "*.psd",
    "*.kra",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_locked_archive(*, url: str, filename: str, sha512: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / filename
    expected = base64.b64decode(sha512)
    if archive.is_file() and hashlib.sha512(archive.read_bytes()).digest() == expected:
        return archive

    request = Request(url, headers={"User-Agent": "Ghostline-Web-Builder/1.0"})
    with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(delete=False) as temporary:
        shutil.copyfileobj(response, temporary)
        temporary_path = Path(temporary.name)
    try:
        payload = temporary_path.read_bytes()
        if hashlib.sha512(payload).digest() != expected:
            raise RuntimeError(f"Locked web dependency {filename} failed its SHA-512 check")
        shutil.move(str(temporary_path), archive)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive


def _download_locked_file(*, url: str, filename: str, sha512: str) -> Path:
    """Download one immutable release document with the same gate as npm archives."""

    return _download_locked_archive(url=url, filename=filename, sha512=sha512)


def _stage_ort_runtime(static: Path) -> None:
    archive = _download_locked_archive(
        url=ORT_TARBALL,
        filename=f"onnxruntime-web-{ORT_VERSION}.tgz",
        sha512=ORT_SHA512,
    )
    destination = static / ORT_WEB_ROOT
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {f"package/{name}": Path(name).name for name in ORT_FILES}
    with tarfile.open(archive, "r:gz") as package:
        members = {member.name: member for member in package.getmembers()}
        missing = sorted(set(wanted) - set(members))
        if missing:
            raise RuntimeError(f"Locked ONNX Runtime package is missing: {', '.join(missing)}")
        for member_name, output_name in wanted.items():
            source = package.extractfile(members[member_name])
            if source is None:
                raise RuntimeError(f"Could not read {member_name} from ONNX Runtime package")
            with (destination / output_name).open("wb") as target:
                shutil.copyfileobj(source, target)
    license_file = _download_locked_file(
        url=ORT_LICENSE_URL,
        filename=f"onnxruntime-web-{ORT_VERSION}-LICENSE.txt",
        sha512=ORT_LICENSE_SHA512,
    )
    notices_file = _download_locked_file(
        url=ORT_NOTICES_URL,
        filename=f"onnxruntime-web-{ORT_VERSION}-ThirdPartyNotices.txt",
        sha512=ORT_NOTICES_SHA512,
    )
    for source, relative in (
        (license_file, ORT_LICENSE_PATH),
        (notices_file, ORT_NOTICES_PATH),
    ):
        target = static / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _stage_browserfs(static: Path) -> None:
    archive = _download_locked_archive(
        url=BROWSERFS_TARBALL,
        filename=f"browserfs-{BROWSERFS_VERSION}.tgz",
        sha512=BROWSERFS_SHA512,
    )
    destination = static / BROWSERFS_WEB_PATH
    license_destination = static / BROWSERFS_LICENSE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    license_destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        members = (
            ("package/dist/browserfs.min.js", destination),
            ("package/LICENSE", license_destination),
        )
        for member_name, target_path in members:
            try:
                member = package.getmember(member_name)
            except KeyError as error:
                raise RuntimeError(f"Locked BrowserFS package is missing {member_name}") from error
            source = package.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member_name} from BrowserFS package")
            with target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))


def _write_favicon(path: Path) -> None:
    """Write a dependency-free 32 px neon-line favicon into the build stage."""
    width = height = 32
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            edge = x in (6, 7, 24, 25) or y in (6, 7, 24, 25)
            diagonal = 9 <= x <= 22 and abs((x + y) - 31) <= 1
            if diagonal:
                rows.extend((112, 246, 255, 255))
            elif edge:
                rows.extend((255, 74, 164, 255))
            else:
                rows.extend((5, 9, 18, 255))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _png_chunk(b"IEND", b"")
    )


def _resolve_model(requested: Path | None) -> Path | None:
    candidates = [requested] if requested else [ROOT / "models" / "ghostline-policy.onnx"]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().resolve().is_file():
            return candidate.expanduser().resolve()
    return None


def _current_environment_fingerprint() -> str:
    """Match the source fingerprint stored by imitation collection/export."""

    return environment_fingerprint(ROOT / "src" / "ghostline")


def _onnx_policy_contract(model: Path) -> tuple[int, dict[str, list[int]], dict[str, str]]:
    """Read and validate the recurrent browser contract from ONNX metadata."""

    contract = validate_onnx_policy(
        model,
        expected_fingerprint=_current_environment_fingerprint(),
    )
    return contract.recurrent_size, contract.input_shapes, contract.metadata


def _write_policy_manifest(static: Path, model: Path | None) -> dict[str, object]:
    model_hash = _sha256(model) if model is not None else None
    model_relative = MODEL_DIRECTORY / f"ghostline-policy-{model_hash[:12]}.onnx" if model_hash else None
    destination = static / model_relative if model_relative is not None else None
    if model is not None:
        assert destination is not None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, destination)
        hidden_size, input_shapes, model_metadata = _onnx_policy_contract(model)
    else:
        hidden_size = None
        input_shapes = {**POLICY_INPUT_SHAPES, "hidden": None}
        model_metadata = {}
    manifest: dict[str, object] = {
        "schema": 1,
        "available": model is not None,
        "runtime": f"onnxruntime-web@{ORT_VERSION}",
        "model_url": model_relative.as_posix() if model_relative is not None else None,
        "bytes": destination.stat().st_size if destination is not None and destination.is_file() else 0,
        "sha256": model_hash,
        "hidden_size": hidden_size,
        "model_metadata": model_metadata,
        "decision_hz": 10,
        "inputs": input_shapes,
    }
    (static / "policy-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _runtime_asset_paths(root: Path = ROOT) -> list[Path]:
    """Return only the MIT runtime derivatives declared by the asset manifest."""

    manifest_path = root / "assets" / "licenses.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        distribution = manifest["runtime_distribution"]
        relative_files = distribution["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid Ghostline asset manifest: {manifest_path}") from error
    if manifest.get("project") != "Ghostline" or distribution.get("license") != "MIT":
        raise RuntimeError("web runtime assets must be declared as Ghostline MIT assets")
    if not isinstance(relative_files, list) or not relative_files:
        raise RuntimeError("asset manifest runtime_distribution.files must be a non-empty list")
    disclosed = {
        details["runtime_file"]
        for details in manifest.get("visual_assets", {}).values()
        if isinstance(details, dict) and isinstance(details.get("runtime_file"), str)
    }
    if disclosed and disclosed != set(relative_files):
        raise RuntimeError("asset runtime list does not match disclosed runtime_file records")

    assets_root = (root / "assets").resolve()
    selected: list[Path] = []
    for value in relative_files:
        if not isinstance(value, str):
            raise RuntimeError("asset manifest runtime file entries must be strings")
        source = (root / value).resolve()
        try:
            source.relative_to(assets_root)
        except ValueError as error:
            raise RuntimeError(f"web runtime asset escapes assets/: {value}") from error
        lowered = source.name.casefold()
        parts = {part.casefold() for part in Path(value).parts}
        if (
            "-source." in lowered
            or "-source-" in lowered
            or "-web." in lowered
            or "screenshots" in parts
            or lowered == "ghostline-key-art-menu.png"
        ):
            raise RuntimeError(f"source/portfolio art cannot be a web runtime asset: {value}")
        if not source.is_file():
            raise RuntimeError(f"declared web runtime asset does not exist: {source}")
        selected.append(source)
    relative_selected = [path.relative_to(root).as_posix() for path in selected]
    if len(set(relative_selected)) != len(relative_selected):
        raise RuntimeError("asset manifest contains duplicate web runtime files")
    return selected


def _stage_runtime_source(stage_root: Path, root: Path = ROOT) -> None:
    package_source = root / "src" / "ghostline"
    package_target = stage_root / "ghostline"
    package_target.mkdir(parents=True, exist_ok=True)
    for name in WEB_RUNTIME_MODULES:
        source = package_source / name
        if not source.is_file():
            raise RuntimeError(f"web runtime allowlist points to a missing module: {source}")
        shutil.copy2(source, package_target / name)

    asset_manifest = root / "assets" / "licenses.json"
    asset_manifest_target = stage_root / "assets" / "licenses.json"
    asset_manifest_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset_manifest, asset_manifest_target)
    for source in _runtime_asset_paths(root):
        relative = source.relative_to(root)
        destination = stage_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def stage(*, model: Path | None = None, include_ort: bool = True) -> dict[str, object]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    _stage_runtime_source(STAGE)
    shutil.copy2(ROOT / "web" / "main.py", STAGE / "main.py")
    shutil.copy2(ROOT / "web" / "runtime.py", STAGE / "web_runtime.py")
    shutil.copy2(ROOT / "LICENSE", STAGE / "LICENSE")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", STAGE / "THIRD_PARTY_NOTICES.md")

    static = STAGE / "static"
    static.mkdir()
    shutil.copy2(ROOT / "LICENSE", static / "LICENSE")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", static / "THIRD_PARTY_NOTICES.md")
    for source in ("ghostline-shell.mjs", "matched-runs.mjs", "policy-bridge.mjs", "ghostline.css"):
        shutil.copy2(ROOT / "web" / "static" / source, static / source)
    _stage_browserfs(static)
    if include_ort:
        _stage_ort_runtime(static)
    _write_favicon(STAGE / "favicon.png")
    return _write_policy_manifest(static, model)


def _payload_size(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def bundle_report(output: Path = OUTPUT) -> dict[str, object]:
    if not output.is_dir():
        raise RuntimeError(f"Web output does not exist: {output}")
    required = (
        output / "index.html",
        output / "ghostline.tar.gz",
        output / "ghostline-shell.mjs",
        output / "matched-runs.mjs",
        output / "policy-bridge.mjs",
        output / "policy-manifest.json",
        output / "ghostline.css",
        output / "LICENSE",
        output / "THIRD_PARTY_NOTICES.md",
        output / BROWSERFS_LICENSE_PATH,
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete web build; missing: {', '.join(missing)}")
    unused = [name for name in ("ghostline.apk",) if (output / name).is_file()]
    if unused:
        raise RuntimeError(f"Web build contains unused Pygbag artifact(s): {', '.join(unused)}")

    browserfs = output / BROWSERFS_WEB_PATH
    if not browserfs.is_file():
        raise RuntimeError(f"Incomplete web build; missing: {BROWSERFS_WEB_PATH.as_posix()}")
    human_files = [*required, output / "favicon.png", browserfs]
    wasm_files = [
        output / ORT_WEB_ROOT / "ort.all.min.mjs",
        output / ORT_WEB_ROOT / "ort-wasm-simd-threaded.mjs",
        output / ORT_WEB_ROOT / "ort-wasm-simd-threaded.wasm",
    ]
    webgpu_files = [
        output / ORT_WEB_ROOT / "ort-wasm-simd-threaded.jsep.mjs",
        output / ORT_WEB_ROOT / "ort-wasm-simd-threaded.jsep.wasm",
    ]
    ort_documents = [output / ORT_LICENSE_PATH, output / ORT_NOTICES_PATH]
    manifest = json.loads((output / "policy-manifest.json").read_text(encoding="utf-8"))
    model = output / str(manifest["model_url"]) if manifest.get("model_url") else output / "models" / "unavailable.onnx"
    if manifest.get("available"):
        runtime_files = [*wasm_files, *ort_documents]
        runtime_missing = [path.relative_to(output).as_posix() for path in runtime_files if not path.is_file()]
        if runtime_missing:
            raise RuntimeError(
                f"Agent build is missing WASM runtime/license files: {', '.join(runtime_missing)}"
            )
        if not model.is_file():
            raise RuntimeError(f"Policy manifest points to a missing model: {manifest.get('model_url')}")
    legal_paths = [output / "THIRD_PARTY_NOTICES.md", output / BROWSERFS_LICENSE_PATH]
    if manifest.get("available"):
        legal_paths.extend(ort_documents)
    report: dict[str, object] = {
        "schema": 1,
        "pygbag_version": PYGBAG_VERSION,
        "onnxruntime_web_version": ORT_VERSION,
        "browserfs_version": BROWSERFS_VERSION,
        "model_available": bool(manifest["available"]),
        "human_first_run_bytes_local": _payload_size(human_files),
        "wasm_agent_total_bytes_local": _payload_size([*human_files, *wasm_files, *ort_documents, model]),
        "webgpu_agent_total_bytes_local": _payload_size(
            [*human_files, *wasm_files, *webgpu_files, *ort_documents, model]
        ),
        "deployment_bytes": _payload_size(output.rglob("*")),
        "model_bytes": model.stat().st_size if model.is_file() else 0,
        "model_sha256": _sha256(model) if model.is_file() else None,
        "runtime_module_allowlist": list(WEB_RUNTIME_MODULES),
        "runtime_asset_allowlist": [
            path.relative_to(ROOT).as_posix() for path in _runtime_asset_paths(ROOT)
        ],
        "legal_documents": [
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in legal_paths
        ],
    }
    return report


def validate_bundle(
    output: Path = OUTPUT,
    *,
    initial_budget: int = DEFAULT_INITIAL_BUDGET,
    wasm_agent_budget: int = DEFAULT_WASM_AGENT_BUDGET,
    require_model: bool = False,
) -> dict[str, object]:
    report = bundle_report(output)
    if require_model and not report["model_available"]:
        raise RuntimeError("portfolio web bundle requires the selected champion ONNX policy")
    if int(report["human_first_run_bytes_local"]) > initial_budget:
        raise RuntimeError(
            f"Human first-run payload is {report['human_first_run_bytes_local']} bytes; budget is {initial_budget}"
        )
    if report["model_available"] and int(report["wasm_agent_total_bytes_local"]) > wasm_agent_budget:
        raise RuntimeError(
            f"WASM agent payload is {report['wasm_agent_total_bytes_local']} bytes; budget is {wasm_agent_budget}"
        )
    (output / "bundle-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build(
    *,
    serve: bool = False,
    model: Path | None = None,
    strict_model: bool = True,
    human_only: bool = False,
    initial_budget_mb: float = 25.0,
    wasm_agent_budget_mb: float = 35.0,
) -> int:
    if human_only and model is not None:
        print("error: --human-only cannot be combined with --model", file=sys.stderr)
        return 2
    resolved_model = None if human_only else _resolve_model(model)
    if not human_only and resolved_model is None:
        print(
            "error: portfolio web builds require --model or models/ghostline-policy.onnx; "
            "use --human-only only for a diagnostic build",
            file=sys.stderr,
        )
        return 2
    manifest = stage(model=resolved_model, include_ort=resolved_model is not None)
    command = [
        sys.executable,
        "-m",
        "pygbag",
        "--title",
        "Ghostline // Procedural Stealth",
        "--width",
        "1280",
        "--height",
        "720",
        "--template",
        str(ROOT / "web" / "ghostline.tmpl"),
        "--icon",
        str(STAGE / "favicon.png"),
    ]
    if not serve:
        command.append("--build")
    command.append(str(STAGE))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    if serve:
        return 0
    # Pygbag emits an itch/Android-style archive in addition to the browser
    # tarball. Vercel never references it, so retaining it almost doubles the
    # static deployment without reducing startup time.
    (OUTPUT / "ghostline.apk").unlink(missing_ok=True)
    shutil.copy2(ROOT / "LICENSE", OUTPUT / "LICENSE")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", OUTPUT / "THIRD_PARTY_NOTICES.md")
    try:
        report = validate_bundle(
            initial_budget=int(initial_budget_mb * 1024 * 1024),
            wasm_agent_budget=int(wasm_agent_budget_mb * 1024 * 1024),
            require_model=not human_only,
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(OUTPUT), "policy": manifest, "bundle": report}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the production Ghostline Pygbag/Vercel bundle")
    parser.add_argument("--serve", action="store_true", help="Run Pygbag's local development server")
    parser.add_argument("--model", type=Path, help="ONNX policy to lazy-load in Agent Lab")
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="Compatibility flag; champion policy validation is already the default",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="Build an explicit diagnostic without ONNX Runtime or a policy (not releasable)",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate the existing output without rebuilding")
    parser.add_argument("--initial-budget-mb", type=float, default=25.0)
    parser.add_argument("--wasm-agent-budget-mb", type=float, default=35.0)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    if arguments.check_only:
        try:
            print(
                json.dumps(
                    validate_bundle(
                        initial_budget=int(arguments.initial_budget_mb * 1024 * 1024),
                        wasm_agent_budget=int(arguments.wasm_agent_budget_mb * 1024 * 1024),
                        require_model=not arguments.human_only,
                    ),
                    indent=2,
                )
            )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(1)
        raise SystemExit(0)
    raise SystemExit(
        build(
            serve=arguments.serve,
            model=arguments.model,
            strict_model=True,
            human_only=arguments.human_only,
            initial_budget_mb=arguments.initial_budget_mb,
            wasm_agent_budget_mb=arguments.wasm_agent_budget_mb,
        )
    )
