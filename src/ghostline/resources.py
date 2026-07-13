"""Runtime asset lookup that works from source, wheels, and PyInstaller."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
import sys
from typing import Iterator


@contextmanager
def runtime_asset_path(relative: str | Path) -> Iterator[Path | None]:
    """Yield an existing runtime asset path, or ``None`` when unavailable.

    Installed distributions keep reviewed assets inside ``ghostline/_assets``.
    Source checkouts and the one-file Windows player retain their conventional
    top-level ``assets`` directory.  ``as_file`` also keeps this safe for any
    importer that does not expose package resources as ordinary files.
    """

    relative = Path(relative)
    packaged = files("ghostline").joinpath("_assets", *relative.parts)
    try:
        if packaged.is_file():
            with as_file(packaged) as resolved:
                yield Path(resolved)
                return
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        pass

    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / relative)
    candidates.extend(
        (
            Path(__file__).resolve().parents[2] / relative,
            Path.cwd() / relative,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            yield candidate
            return
    yield None
