from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ghostline import __version__
from ghostline.onnx_contract import environment_fingerprint, validate_onnx_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = Path("models/ghostline-policy.onnx")
PLAYER_ENTRY = Path("src/ghostline/player_entry.py")
ASSET_MANIFEST = Path("assets/licenses.json")
THIRD_PARTY_NOTICES = Path("THIRD_PARTY_NOTICES.md")
POLICY_PARITY_DOCUMENT = "Ghostline.policy-parity.json"
MIN_RELEASE_PARITY_SAMPLES = 1_000
EXCLUDED_PLAYER_MODULES = (
    "ghostline.ablation",
    "ghostline.exporting",
    "ghostline.imitation",
    "ghostline.model",
    "ghostline.recording",
    "ghostline.rnd",
    "ghostline.torchrl_train",
    "ghostline.training",
    "imageio",
    "imageio_ffmpeg",
    "matplotlib",
    "moviepy",
    "onnx",
    "pandas",
    "PIL",
    "sympy",
    "tensorboard",
    "tqdm",
    "torch",
)
FORBIDDEN_PLAYER_ROOTS = {
    "imageio",
    "imageio_ffmpeg",
    "matplotlib",
    "moviepy",
    "onnx",
    "pandas",
    "pil",
    "tensorboard",
    "tensordict",
    "tqdm",
    "torch",
    "torchrl",
}


def _resolve_from_root(path: Path, root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _data_argument(source: Path, destination: str) -> str:
    """Return PyInstaller's platform-aware SOURCE:DEST data argument."""

    return f"{source}{os.pathsep}{destination}"


def _runtime_asset_paths(root: Path) -> list[Path]:
    """Load and validate the authoritative asset distribution contract."""

    manifest_path = root / ASSET_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        distribution = manifest["runtime_distribution"]
        relative_files = distribution["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid Ghostline asset manifest: {manifest_path}") from error
    if manifest.get("project") != "Ghostline" or distribution.get("license") != "MIT":
        raise ValueError("asset manifest must identify Ghostline runtime assets under MIT")
    if not isinstance(relative_files, list) or not relative_files:
        raise ValueError("asset manifest runtime_distribution.files must be a non-empty list")
    disclosed_runtime_files = {
        details["runtime_file"]
        for details in manifest.get("visual_assets", {}).values()
        if isinstance(details, dict) and isinstance(details.get("runtime_file"), str)
    }
    if disclosed_runtime_files and disclosed_runtime_files != set(relative_files):
        raise ValueError("asset runtime list does not match the disclosed runtime_file records")

    assets_root = (root / "assets").resolve()
    selected: list[Path] = []
    for value in relative_files:
        if not isinstance(value, str):
            raise ValueError("asset manifest runtime file entries must be strings")
        relative = Path(value)
        source = (root / relative).resolve()
        try:
            source.relative_to(assets_root)
        except ValueError as error:
            raise ValueError(f"runtime asset escapes assets/: {value}") from error
        lowered = source.name.casefold()
        if "-source." in lowered or "-source-" in lowered or "-web." in lowered:
            raise ValueError(f"source/web provenance file cannot be a runtime asset: {value}")
        if "screenshots" in {part.casefold() for part in relative.parts} or lowered == "ghostline-key-art-menu.png":
            raise ValueError(f"portfolio/retired art cannot be a runtime asset: {value}")
        if not source.is_file():
            raise ValueError(f"declared runtime asset does not exist: {source}")
        selected.append(source)
    if len({path.relative_to(root).as_posix() for path in selected}) != len(selected):
        raise ValueError("asset manifest contains duplicate runtime files")
    return selected


def _release_assets(root: Path) -> list[tuple[Path, str]]:
    assets = (root / "assets").resolve()
    sources = [root / ASSET_MANIFEST, *_runtime_asset_paths(root)]
    selected: list[tuple[Path, str]] = []
    for source in sources:
        relative_parent = source.resolve().relative_to(assets).parent
        destination = (Path("assets") / relative_parent).as_posix()
        selected.append((source, destination))
    return selected


def windows_build_command(
    *,
    root: Path = PROJECT_ROOT,
    model: Path | None = DEFAULT_POLICY,
) -> list[str]:
    """Compose the deterministic player-only PyInstaller command.

    The desktop executable intentionally starts from ``player_entry.py`` rather
    than the developer CLI. This keeps training, media, and PyTorch modules out
    of the player dependency graph while preserving ONNX Runtime inference.
    """

    root = root.resolve()
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "Ghostline",
        "--paths",
        str(root / "src"),
    ]
    for source, destination in _release_assets(root):
        command.extend(("--add-data", _data_argument(source, destination)))
    command.extend(("--add-data", _data_argument(root / "LICENSE", ".")))
    command.extend(("--add-data", _data_argument(root / THIRD_PARTY_NOTICES, ".")))
    if model is not None:
        resolved_model = _resolve_from_root(model, root)
        command.extend(
            [
                "--add-data",
                _data_argument(resolved_model, "models"),
                "--hidden-import",
                "ghostline.inference",
                "--hidden-import",
                "ghostline.env",
            ]
        )
    else:
        command.extend(("--exclude-module", "ghostline.inference"))
        command.extend(("--exclude-module", "onnxruntime"))
    for module in EXCLUDED_PLAYER_MODULES:
        command.extend(("--exclude-module", module))
    command.append(str(root / PLAYER_ENTRY))
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_policy_evidence(
    root: Path,
    model: Path,
) -> tuple[dict[str, object], Path]:
    """Bind a player policy to its v2 metadata and recurrent parity evidence."""

    expected_fingerprint = environment_fingerprint(root / "src" / "ghostline")
    contract = validate_onnx_policy(model, expected_fingerprint=expected_fingerprint)
    model_sha256 = _sha256(model)
    exact = model.with_suffix(".parity.json")
    conventional = model.with_name(f"{model.stem}.fp32.parity.json")
    candidates: list[Path] = []
    for candidate in (exact, conventional, *sorted(model.parent.glob("*.parity.json"))):
        resolved = candidate.resolve()
        if resolved not in candidates and resolved.is_file():
            candidates.append(resolved)
    if not candidates:
        raise RuntimeError(
            f"portfolio policy has no export parity report beside {model.name}; "
            "run `ghostline export ... --parity-samples 1000 --deployment-output ...`"
        )

    failures: list[str] = []
    for report_path in candidates:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("root is not an object")
            if report.get("status") != "passed":
                raise ValueError("status is not passed")
            if report.get("observation_contract") != "GhostlineEnv-v2":
                raise ValueError("observation contract is not GhostlineEnv-v2")
            if report.get("environment_fingerprint") != expected_fingerprint:
                raise ValueError("environment fingerprint is stale")
            checkpoint_sha256 = str(report.get("checkpoint_sha256", ""))
            if re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha256) is None:
                raise ValueError("checkpoint SHA-256 is missing or malformed")
            artifacts = report.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError("artifact records are missing")
            matching = [
                (name, record)
                for name, record in artifacts.items()
                if isinstance(record, dict) and record.get("sha256") == model_sha256
            ]
            if not matching:
                raise ValueError("no audited ONNX artifact matches the packaged bytes")
            artifact_name, artifact = matching[0]
            parity = artifact.get("parity")
            if not isinstance(parity, dict):
                raise ValueError("matching artifact has no recurrent parity record")
            samples = int(parity.get("samples", 0))
            mismatches = int(parity.get("action_mismatches", -1))
            horizon = int(parity.get("sequence_horizon", 0))
            tiers = sorted({int(value) for value in parity.get("tiers", [])})
            if parity.get("passed") is not True or samples < MIN_RELEASE_PARITY_SAMPLES or mismatches != 0:
                raise ValueError(
                    f"recurrent parity requires >= {MIN_RELEASE_PARITY_SAMPLES} samples and zero mismatches"
                )
            if horizon < 128 or tiers != [1, 2, 3, 4, 5, 6]:
                raise ValueError("recurrent parity does not cover 128-step sequences across all tiers")
            top_samples = int(report.get("parity_samples", 0))
            top_mismatches = int(report.get("mismatches", -1))
            if top_samples < MIN_RELEASE_PARITY_SAMPLES or top_mismatches != 0:
                raise ValueError("top-level parity summary does not pass the release gate")
            evidence: dict[str, object] = {
                "contract": "GhostlineEnv-v2",
                "environment_fingerprint": expected_fingerprint,
                "recurrent_size": int(contract.recurrent_size),
                "onnx_sha256": model_sha256,
                "checkpoint_sha256": checkpoint_sha256.lower(),
                "precision": str(artifact.get("precision", artifact_name)),
                "parity_samples": samples,
                "action_mismatches": mismatches,
                "sequence_horizon": horizon,
                "tiers": tiers,
                "parity_report_file": POLICY_PARITY_DOCUMENT,
                "parity_report_sha256": _sha256(report_path),
            }
            return evidence, report_path
        except (OSError, json.JSONDecodeError, OverflowError, TypeError, ValueError) as error:
            failures.append(f"{report_path.name}: {error}")
    raise RuntimeError(
        "no export parity report authorizes the selected ONNX policy: " + "; ".join(failures)
    )


def _file_record(path: Path, *, name: str | None = None) -> dict[str, object]:
    return {
        "file": name or path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_manifest(
    root: Path,
    executable: Path,
    model: Path | None,
    *,
    release_documents: list[Path] | None = None,
    runtime_packages: list[dict[str, str]] | None = None,
    policy_validation: dict[str, object] | None = None,
) -> Path:
    license_file = root / "LICENSE"
    asset_manifest = root / ASSET_MANIFEST
    runtime_assets = _runtime_asset_paths(root)
    distribution_root = root / "dist"
    if model is not None and policy_validation is None:
        raise ValueError("portfolio package manifest requires policy validation evidence")
    if model is not None and policy_validation.get("onnx_sha256") != _sha256(model):
        raise ValueError("policy validation evidence does not match the bundled ONNX bytes")
    manifest = {
        "schema": 3,
        "product": "Ghostline",
        "version": __version__,
        "executable": executable.name,
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": _sha256(executable),
        "policy_bundled": model is not None,
        "policy_file": DEFAULT_POLICY.as_posix() if model is not None else None,
        "policy_bytes": model.stat().st_size if model is not None else None,
        "policy_sha256": _sha256(model) if model is not None else None,
        "policy_validation": policy_validation,
        "license": "MIT",
        "license_file": "LICENSE",
        "license_sha256": _sha256(license_file),
        "asset_manifest": _file_record(asset_manifest, name="ASSET-LICENSES.json"),
        "runtime_assets": [
            _file_record(path, name=path.relative_to(root).as_posix())
            for path in runtime_assets
        ],
        "release_documents": [
            _file_record(path, name=path.relative_to(distribution_root).as_posix())
            for path in (release_documents or [])
        ],
        "runtime_packages": runtime_packages or [],
        "user_data": {
            "root": "%LOCALAPPDATA%/Ghostline",
            "progression": "progression-v1.json",
            "run_telemetry": "runs-v1.jsonl",
        },
        "recording": {
            "run_telemetry_enabled": True,
            "video_recorder_bundled": False,
            "video_recorder_install_extra": "media",
        },
        "pytorch_bundled": False,
        "forbidden_package_roots_verified_absent": sorted(FORBIDDEN_PLAYER_ROOTS),
    }
    output = distribution_root / "Ghostline.manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def _canonical_policy_source(root: Path, model: Path) -> Path:
    """Give every selected checkpoint the runtime's stable bundled filename."""

    if model.name == DEFAULT_POLICY.name:
        return model
    staged = root / "build" / "Ghostline-package" / DEFAULT_POLICY.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, staged)
    return staged


def _verify_player_archive(executable: Path) -> set[str]:
    """Fail closed if developer-only ML/media packages entered the player."""

    completed = subprocess.run(
        [sys.executable, "-m", "PyInstaller.utils.cliutils.archive_viewer", "-r", str(executable)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError("could not inspect the PyInstaller archive")
    archive_roots: set[str] = set()
    for line in completed.stdout.splitlines():
        if "'" not in line:
            continue
        entry = line.rsplit("'", 2)[-2].replace("\\", "/")
        root = re.split(r"[./]", entry, maxsplit=1)[0].casefold()
        if root:
            archive_roots.add(root)
    forbidden = archive_roots & FORBIDDEN_PLAYER_ROOTS
    if forbidden:
        raise RuntimeError(f"player archive contains forbidden packages: {', '.join(sorted(forbidden))}")
    return archive_roots


def _archive_distribution_names(archive_roots: set[str]) -> list[str]:
    package_map = {key.casefold(): values for key, values in metadata.packages_distributions().items()}
    names = {
        distribution_name
        for root in archive_roots
        for distribution_name in package_map.get(root, [])
        if distribution_name.casefold() != "ghostline"
    }
    # PyInstaller's bootloader is part of the executable but is not an import
    # root in the archive. Its special-exception license still ships beside it.
    names.add("pyinstaller")
    return sorted(names, key=str.casefold)


def _declared_license(distribution: metadata.Distribution) -> str:
    expression = distribution.metadata.get("License-Expression")
    if expression:
        return expression
    declared = distribution.metadata.get("License")
    if declared:
        return declared
    classifiers = distribution.metadata.get_all("Classifier", [])
    license_classifiers = [value.rsplit("::", 1)[-1].strip() for value in classifiers if value.startswith("License ::")]
    return "; ".join(license_classifiers) or "See bundled license files"


def _runtime_package_records(archive_roots: set[str]) -> list[dict[str, str]]:
    records = [
        {
            "name": "Python",
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "license": "Python Software Foundation License",
        }
    ]
    for name in _archive_distribution_names(archive_roots):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        records.append(
            {
                "name": distribution.metadata.get("Name", name),
                "version": distribution.version,
                "license": _declared_license(distribution),
            }
        )
    return records


def _runtime_license_sources(archive_roots: set[str]) -> list[tuple[Path, Path]]:
    """Locate exact installed license texts for discovered binary contents."""

    selected: dict[str, tuple[Path, Path]] = {}
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        selected["python/LICENSE.txt"] = (python_license, Path("python/LICENSE.txt"))

    for name in _archive_distribution_names(archive_roots):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", distribution.metadata.get("Name", name)).strip("-")
        for entry in distribution.files or []:
            lowered_name = entry.name.casefold()
            is_notice_name = any(token in lowered_name for token in ("license", "copying", "notice"))
            if not is_notice_name or Path(entry.name).suffix.casefold() not in ("", ".txt", ".md", ".rst"):
                continue
            source = Path(distribution.locate_file(entry))
            if not source.is_file():
                continue
            relative = Path(*entry.parts)
            destination = Path(safe_name) / relative
            selected[destination.as_posix()] = (source, destination)

        # pygame-ce 2.5.x installs its full LGPL text as generated docs but
        # does not list that file in wheel metadata on every platform.
        if safe_name.casefold() == "pygame-ce":
            fallback = Path(distribution.locate_file("pygame/docs/generated/LGPL.txt"))
            if fallback.is_file():
                destination = Path(safe_name) / "LGPL-2.1.txt"
                selected[destination.as_posix()] = (fallback, destination)
    return [selected[key] for key in sorted(selected)]


def _copy_release_documents(
    root: Path,
    archive_roots: set[str],
    *,
    policy_parity_report: Path | None = None,
) -> list[Path]:
    distribution_root = (root / "dist").resolve()
    licenses_root = (distribution_root / "licenses").resolve()
    if licenses_root.parent != distribution_root:
        raise RuntimeError("refusing to write release licenses outside dist/")
    if licenses_root.exists():
        shutil.rmtree(licenses_root)
    licenses_root.mkdir(parents=True)

    documents = [
        (root / "LICENSE", distribution_root / "LICENSE"),
        (root / THIRD_PARTY_NOTICES, distribution_root / THIRD_PARTY_NOTICES.name),
        (root / ASSET_MANIFEST, distribution_root / "ASSET-LICENSES.json"),
    ]
    if policy_parity_report is not None:
        documents.append(
            (policy_parity_report, distribution_root / POLICY_PARITY_DOCUMENT)
        )
    copied: list[Path] = []
    for source, destination in documents:
        shutil.copy2(source, destination)
        copied.append(destination)
    for source, relative in _runtime_license_sources(archive_roots):
        destination = (licenses_root / relative).resolve()
        try:
            destination.relative_to(licenses_root)
        except ValueError as error:
            raise RuntimeError(f"refusing unsafe runtime license destination: {relative}") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _verify_player_smoke(executable: Path, *, policy_bundled: bool) -> None:
    command = [str(executable), "--release-smoke-test"]
    if not policy_bundled:
        command.append("--human-only")
    try:
        completed = subprocess.run(command, check=False, timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("packaged player smoke test timed out after 120 seconds") from error
    if completed.returncode:
        raise RuntimeError(f"packaged player smoke test failed with exit code {completed.returncode}")


def build_windows(
    *,
    model: Path | None = DEFAULT_POLICY,
    dry_run: bool = False,
    root: Path = PROJECT_ROOT,
) -> int:
    """Build the Windows player, requiring a selected policy by default."""

    root = root.resolve()
    assets = root / "assets"
    if not assets.is_dir():
        print(f"error: required asset directory does not exist: {assets}", file=sys.stderr)
        return 2
    license_file = root / "LICENSE"
    if not license_file.is_file():
        print(f"error: required MIT license does not exist: {license_file}", file=sys.stderr)
        return 2
    for required in (root / PLAYER_ENTRY, root / THIRD_PARTY_NOTICES, root / ASSET_MANIFEST):
        if not required.is_file():
            print(f"error: required release input does not exist: {required}", file=sys.stderr)
            return 2
    try:
        _runtime_asset_paths(root)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    resolved_model = _resolve_from_root(model, root) if model is not None else None
    if resolved_model is not None and not resolved_model.is_file():
        print(
            f"error: selected ONNX policy does not exist: {resolved_model}\n"
            "Use --human-only only for a diagnostic build without Agent Lab inference.",
            file=sys.stderr,
        )
        return 2
    policy_validation: dict[str, object] | None = None
    policy_parity_report: Path | None = None
    if resolved_model is not None:
        try:
            policy_validation, policy_parity_report = _release_policy_evidence(root, resolved_model)
        except RuntimeError as error:
            print(f"error: selected ONNX policy failed the portfolio release gate: {error}", file=sys.stderr)
            return 2
    bundled_model = _canonical_policy_source(root, resolved_model) if resolved_model is not None else None
    command = windows_build_command(root=root, model=bundled_model)
    print(subprocess.list2cmdline(command))
    if dry_run:
        return 0
    if sys.platform != "win32":
        print("error: the desktop player build must run on Windows", file=sys.stderr)
        return 2
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        return completed.returncode
    executable = root / "dist" / "Ghostline.exe"
    if not executable.is_file():
        print(f"error: PyInstaller did not produce {executable}", file=sys.stderr)
        return 1
    try:
        archive_roots = _verify_player_archive(executable)
        _verify_player_smoke(executable, policy_bundled=bundled_model is not None)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        release_documents = _copy_release_documents(
            root,
            archive_roots,
            policy_parity_report=policy_parity_report,
        )
        manifest = _write_manifest(
            root,
            executable,
            bundled_model,
            release_documents=release_documents,
            runtime_packages=_runtime_package_records(archive_roots),
            policy_validation=policy_validation,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: could not finalize release documents: {error}", file=sys.stderr)
        return 1
    print(f"release manifest: {manifest}")
    print(f"release documents: {len(release_documents)} files under {root / 'dist'}")
    return 0
