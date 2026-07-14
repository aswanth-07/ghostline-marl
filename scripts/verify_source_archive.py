"""Audit a Ghostline source distribution without extracting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "AGENTS.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "requirements.lock",
    "setup.py",
    "vercel.json",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    "assets/licenses.json",
    "assets/screenshots/gameplay-stealth-v3.png",
    "assets/visual/ghostline-environment-atlas-source-v1.png",
    "assets/visual/ghostline-environment-atlas-v1.png",
    "benchmarks/final-test-slices.json",
    "benchmarks/teacher/audit-history.json",
    "models/model-card.md",
    "scripts/benchmark_ghostline.py",
    "scripts/build_web.py",
    "scripts/verify_release_evidence.py",
    "scripts/verify_source_archive.py",
    "src/ghostline/simulation.py",
    "tests/test_ghostline.py",
    "tests/test_release_evidence.py",
    "web/ghostline.tmpl",
    "web/main.py",
    "web/static/policy-bridge.mjs",
    "web/tests/policy-bridge.test.mjs",
    "wiki/setup.md",
    "wiki/training.md",
    "wiki/web-deployment.md",
}
FORBIDDEN = {
    "tests/test_cli.py",
    "tests/test_env.py",
    "tests/test_training.py",
}
FORBIDDEN_PREFIXES = ("src/neon_arena/",)
RELEASE_REQUIRED = {
    "benchmarks/neural/champion-final-8m-500.json",
    "benchmarks/neural/champion-final-8m-500.csv",
    "benchmarks/neural/champion-final-8m-500.episodes.csv",
    "benchmarks/neural/champion-onnx-parity.json",
    "benchmarks/system/headless-throughput.json",
}


def _relative_members(archive: tarfile.TarFile) -> set[str]:
    names: set[str] = set()
    roots: set[str] = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"source archive has an unsafe member path: {member.name}")
        if len(path.parts) < 2:
            continue
        roots.add(path.parts[0])
        names.add(PurePosixPath(*path.parts[1:]).as_posix())
    if len(roots) != 1:
        raise RuntimeError("source archive must have exactly one top-level project directory")
    return names


def verify(path: Path, *, release: bool = False) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source archive does not exist: {path}")
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = _relative_members(archive)
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"could not inspect source archive: {path}") from error
    required = REQUIRED | (RELEASE_REQUIRED if release else set())
    missing = sorted(required - members)
    if missing:
        raise RuntimeError("source archive is incomplete; missing: " + ", ".join(missing))
    forbidden = sorted(FORBIDDEN & members)
    forbidden.extend(
        sorted(name for name in members if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES))
    )
    forbidden.extend(sorted(name for name in members if name.endswith((".pyc", ".pyo"))))
    if forbidden:
        raise RuntimeError("source archive contains excluded legacy/generated files: " + ", ".join(forbidden))
    return {
        "status": "passed",
        "archive": str(path),
        "members": len(members),
        "required_files": len(required),
        "release_evidence_required": release,
        "legacy_test_modules_excluded": sorted(FORBIDDEN),
        "legacy_package_excluded": True,
    }


def _default_archive() -> Path:
    candidates = sorted(
        (ROOT / "dist").glob("ghostline-*.tar.gz"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("no Ghostline sdist under dist/; run `python -m build` first")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", help="Defaults to newest dist/ghostline-*.tar.gz")
    parser.add_argument(
        "--release",
        action="store_true",
        help="Also require the canonical final neural/parity/throughput evidence.",
    )
    args = parser.parse_args()
    try:
        report = verify(args.archive or _default_archive(), release=args.release)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"source archive failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    main()
