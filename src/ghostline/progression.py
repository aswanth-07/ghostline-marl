from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_BINDINGS: dict[str, str] = {
    "move_up": "w",
    "move_down": "s",
    "move_left": "a",
    "move_right": "d",
    "dash": "left shift",
    "pulse": "space",
    "restart": "r",
    "pause": "escape",
    "menu_up": "up",
    "menu_down": "down",
    "confirm": "return",
    "back": "escape",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "audio": {
        "enabled": True,
        "master": 0.75,
        "music": 0.55,
        "sfx": 0.85,
    },
    "accessibility": {
        "high_contrast": False,
        "color_safe": False,
        "reduced_motion": False,
        "reduced_flashes": True,
        "sound_captions": False,
        "hud_scale": 1.0,
        "timer_assist": False,
        "timer_warnings": True,
        "tutorial_hints": True,
    },
    "display": {
        "screen_shake": True,
        "fullscreen": False,
    },
    "bindings": DEFAULT_BINDINGS,
}


def user_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".ghostline")) / "Ghostline"


def progression_path() -> Path:
    return user_data_dir() / "progression-v1.json"


def telemetry_path() -> Path:
    return user_data_dir() / "runs-v1.jsonl"


def _defaults() -> dict[str, Any]:
    return deepcopy(DEFAULT_SETTINGS)


def _bounded_float(value: object, default: float, *, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_settings(raw: Mapping[str, object] | None) -> dict[str, Any]:
    """Migrate and validate a persisted settings profile without importing Pygame."""

    result = _defaults()
    if not isinstance(raw, Mapping):
        return result

    raw_audio = raw.get("audio")
    if isinstance(raw_audio, Mapping):
        result["audio"]["enabled"] = bool(raw_audio.get("enabled", result["audio"]["enabled"]))
        for key in ("master", "music", "sfx"):
            result["audio"][key] = _bounded_float(raw_audio.get(key), result["audio"][key])

    raw_accessibility = raw.get("accessibility")
    if isinstance(raw_accessibility, Mapping):
        for key in (
            "high_contrast",
            "color_safe",
            "reduced_motion",
            "reduced_flashes",
            "sound_captions",
            "timer_assist",
            "timer_warnings",
            "tutorial_hints",
        ):
            result["accessibility"][key] = bool(
                raw_accessibility.get(key, result["accessibility"][key])
            )
        hud_scale = _bounded_float(raw_accessibility.get("hud_scale"), 1.0, low=1.0, high=1.5)
        result["accessibility"]["hud_scale"] = min((1.0, 1.25, 1.5), key=lambda value: abs(value - hud_scale))

    raw_display = raw.get("display")
    if isinstance(raw_display, Mapping):
        result["display"]["screen_shake"] = bool(
            raw_display.get("screen_shake", result["display"]["screen_shake"])
        )
        result["display"]["fullscreen"] = bool(
            raw_display.get("fullscreen", result["display"]["fullscreen"])
        )

    raw_bindings = raw.get("bindings")
    if isinstance(raw_bindings, Mapping):
        for action, default_name in DEFAULT_BINDINGS.items():
            name = raw_bindings.get(action, default_name)
            if isinstance(name, str) and name.strip():
                result["bindings"][action] = name.strip().lower()[:40]
    return result


def _empty_progression() -> dict[str, object]:
    return {
        "version": 1,
        "highest_unlocked_tier": 1,
        "best_scores": {},
        "settings": _defaults(),
        "recent_runs": [],
    }


def load_progression() -> dict[str, object]:
    path = progression_path()
    if not path.exists():
        return _empty_progression()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_progression()
    if not isinstance(data, Mapping):
        return _empty_progression()
    try:
        highest = int(data.get("highest_unlocked_tier", 1))
    except (TypeError, ValueError):
        highest = 1
    raw_scores = data.get("best_scores", {})
    scores = dict(raw_scores) if isinstance(raw_scores, Mapping) else {}
    recent = data.get("recent_runs", [])
    if not isinstance(recent, list):
        recent = []
    return {
        "version": 1,
        "highest_unlocked_tier": max(1, min(6, highest)),
        "best_scores": scores,
        "settings": normalize_settings(data.get("settings")),
        "recent_runs": [item for item in recent[-20:] if isinstance(item, dict)],
    }


def _save_progression(data: Mapping[str, object]) -> None:
    path = progression_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(data), indent=2), encoding="utf-8")
    temporary.replace(path)


def load_settings() -> dict[str, Any]:
    return normalize_settings(load_progression().get("settings"))


def save_settings(settings: Mapping[str, object]) -> dict[str, Any]:
    data = load_progression()
    normalized = normalize_settings(settings)
    data["settings"] = normalized
    _save_progression(data)
    return normalized


def record_success(*, tier: int, score: int) -> None:
    data = load_progression()
    data["highest_unlocked_tier"] = max(int(data["highest_unlocked_tier"]), min(6, tier + 1))
    scores = dict(data["best_scores"])
    scores[str(tier)] = max(int(scores.get(str(tier), 0)), int(score))
    data["best_scores"] = scores
    _save_progression(data)


def record_run(metrics: Mapping[str, object]) -> dict[str, object]:
    """Append a portfolio benchmark record and keep a compact in-profile history."""

    record = dict(metrics)
    record.setdefault("schema_version", 1)
    record.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    data = load_progression()
    recent = list(data.get("recent_runs", []))
    compact_keys = (
        "schema_version",
        "recorded_at",
        "controller",
        "policy",
        "seed",
        "tier",
        "timer_assistance",
        "success",
        "failure_reason",
        "duration_seconds",
        "detections",
        "damage",
        "max_trace",
        "path_efficiency",
        "policy_latency_mean_ms",
    )
    recent.append({key: record[key] for key in compact_keys if key in record})
    data["recent_runs"] = recent[-20:]
    _save_progression(data)

    path = telemetry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return record
