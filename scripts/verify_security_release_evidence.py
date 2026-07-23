"""Fail-closed audit for Ghostline's adaptive-security release evidence.

The audit binds the selected recurrent MAPPO checkpoint to its two validation
slices and the untouched 13M final-test slice. It also preserves the failed
12M candidate as negative evidence. The verifier never reruns an episode.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Callable

from ghostline.security_model import (
    SECURITY_OBSERVATION_CONTRACT,
    security_environment_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_CONTRACT = "ghostline-security-evaluation-v0"
RUNNER_SHA256 = "76baa30af55cdaa2e71bb6ba06672bd9203455552358017505685827240b2e47"
RUNNER_OPPONENT = f"env-v2:{RUNNER_SHA256}"
TIERS = (3, 4, 5, 6)

SECURITY_CHECKPOINT = Path("models/ghostline-security.pt")
FINAL_REPORT = Path("benchmarks/security/adaptive-security-final-13m-25.json")
FAILED_FINAL_REPORT = Path("benchmarks/security/adaptive-security-final-12m-25.json")
VALIDATION_REPORTS = (
    Path("benchmarks/security/adaptive-security-validation-a-11m-25.json"),
    Path("benchmarks/security/adaptive-security-validation-b-1105m-10.json"),
)
TEACHER_REPORT = Path("benchmarks/security/tactical-strategic-validation-11m-25.json")

FINAL_SEED_START = 13_000_000
FINAL_EPISODES_PER_TIER = 25
FINAL_STOP_COUNTS = {3: 1, 4: 0, 5: 2, 6: 4}


class SecurityReleaseEvidenceError(RuntimeError):
    """Raised when adaptive-security evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SecurityReleaseEvidenceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SecurityReleaseEvidenceError(f"required security evidence is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise SecurityReleaseEvidenceError(f"could not read security evidence: {path}") from error
    if not isinstance(value, dict):
        raise SecurityReleaseEvidenceError(f"security evidence root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        stream = path.open("rb")
    except OSError as error:
        raise SecurityReleaseEvidenceError(f"required security artifact is missing: {path}") from error
    digest = hashlib.sha256()
    with stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _verify_csv_copies(path: Path, report: dict[str, Any]) -> None:
    aggregate_path = path.with_suffix(".csv")
    episode_path = path.with_name(f"{path.stem}.episodes.csv")
    try:
        with aggregate_path.open(encoding="utf-8", newline="") as stream:
            aggregate_rows = list(csv.DictReader(stream))
        with episode_path.open(encoding="utf-8", newline="") as stream:
            episode_rows = list(csv.DictReader(stream))
    except OSError as error:
        raise SecurityReleaseEvidenceError(f"security report CSV is missing beside {path}") from error

    _require(len(aggregate_rows) == len(TIERS), f"aggregate CSV row count differs from {path.name}")
    for row, tier in zip(aggregate_rows, TIERS, strict=True):
        aggregate = report["tiers"][str(tier)]
        _require(int(row["tier"]) == tier, f"aggregate CSV tier order differs from {path.name}")
        for key, expected in aggregate.items():
            actual = int(row[key]) if key == "episodes" else float(row[key])
            _require(
                math.isclose(float(actual), float(expected), abs_tol=1e-12),
                f"aggregate CSV field {key!r} differs from {path.name}",
            )

    records = report["episodes"]
    _require(len(episode_rows) == len(records), f"episode CSV row count differs from {path.name}")
    for ordinal, (row, expected) in enumerate(zip(episode_rows, records, strict=True)):
        converted: dict[str, Any] = {
            "tier": int(row["tier"]),
            "seed": int(row["seed"]),
            "security_stop": row["security_stop"] == "True",
            "runner_success": row["runner_success"] == "True",
            "failure_reason": row["failure_reason"],
            "damage": int(row["damage"]),
            "detections": int(row["detections"]),
            "duration_seconds": float(row["duration_seconds"]),
            "invalid_actions": int(row["invalid_actions"]),
        }
        _require(converted == expected, f"episode CSV row {ordinal} differs from {path.name}")


def _verify_report(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_seed_start: int,
    expected_episodes_per_tier: int,
    expected_checkpoint_sha256: str | None,
    expected_controller: str,
) -> dict[str, Any]:
    report = _read_json(path)
    _require(report.get("contract") == REPORT_CONTRACT, f"unsupported security contract in {path.name}")
    _require(
        report.get("observation_contract") == SECURITY_OBSERVATION_CONTRACT,
        f"wrong observation contract in {path.name}",
    )
    _require(
        report.get("environment_fingerprint") == expected_fingerprint,
        f"stale security environment fingerprint in {path.name}",
    )
    _require(report.get("runner_opponent") == RUNNER_OPPONENT, f"wrong runner opponent in {path.name}")
    _require(report.get("security_controller") == expected_controller, f"wrong controller in {path.name}")
    _require(
        report.get("security_checkpoint_sha256") == expected_checkpoint_sha256,
        f"checkpoint hash differs in {path.name}",
    )
    _require(report.get("seed_start") == expected_seed_start, f"wrong seed slice in {path.name}")
    _require(
        report.get("episodes_per_tier") == expected_episodes_per_tier,
        f"wrong episodes-per-tier in {path.name}",
    )

    records = report.get("episodes")
    _require(isinstance(records, list), f"episode records are missing in {path.name}")
    _require(
        len(records) == expected_episodes_per_tier * len(TIERS),
        f"episode record count is incomplete in {path.name}",
    )
    grouped: dict[int, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    for ordinal, record in enumerate(records):
        _require(isinstance(record, dict), f"episode record {ordinal} is not an object in {path.name}")
        expected_tier = TIERS[ordinal // expected_episodes_per_tier]
        expected_index = ordinal % expected_episodes_per_tier
        _require(record.get("tier") == expected_tier, f"episode tier order differs in {path.name}")
        _require(
            record.get("seed") == expected_seed_start + expected_tier * 100_000 + expected_index,
            f"episode seed formula differs in {path.name}",
        )
        _require(isinstance(record.get("security_stop"), bool), f"security_stop is not boolean in {path.name}")
        _require(isinstance(record.get("runner_success"), bool), f"runner_success is not boolean in {path.name}")
        _require(
            bool(record["security_stop"]) is not bool(record["runner_success"]),
            f"runner/security outcomes disagree in {path.name}",
        )
        _require(isinstance(record.get("failure_reason"), str), f"failure reason is missing in {path.name}")
        for key in ("damage", "detections", "invalid_actions"):
            _require(
                isinstance(record.get(key), int)
                and not isinstance(record.get(key), bool)
                and int(record[key]) >= 0,
                f"episode field {key!r} is invalid in {path.name}",
            )
        _require(
            _is_number(record.get("duration_seconds")) and float(record["duration_seconds"]) > 0.0,
            f"episode duration is invalid in {path.name}",
        )
        grouped[expected_tier].append(record)

    tiers = report.get("tiers")
    _require(
        isinstance(tiers, dict) and set(tiers) == {str(tier) for tier in TIERS},
        f"tier aggregates are incomplete in {path.name}",
    )
    rates: list[float] = []
    for tier, rows in grouped.items():
        aggregate = tiers[str(tier)]
        _require(isinstance(aggregate, dict), f"tier {tier} aggregate is invalid in {path.name}")
        stops = sum(int(row["security_stop"]) for row in rows)
        successes = sum(int(row["runner_success"]) for row in rows)
        expected = {
            "episodes": expected_episodes_per_tier,
            "security_stop_rate": stops / expected_episodes_per_tier,
            "runner_success_rate": successes / expected_episodes_per_tier,
            "mean_damage": fmean(float(row["damage"]) for row in rows),
            "mean_detections": fmean(float(row["detections"]) for row in rows),
            "mean_duration_seconds": fmean(float(row["duration_seconds"]) for row in rows),
        }
        for key, value in expected.items():
            actual = aggregate.get(key)
            _require(
                _is_number(actual) and math.isclose(float(actual), float(value), abs_tol=1e-12),
                f"tier {tier} aggregate {key!r} differs from episodes in {path.name}",
            )
        low, high = _wilson(stops, expected_episodes_per_tier)
        _require(
            _is_number(aggregate.get("security_stop_ci95_low"))
            and math.isclose(float(aggregate["security_stop_ci95_low"]), low, abs_tol=1e-12),
            f"tier {tier} Wilson lower bound differs in {path.name}",
        )
        _require(
            _is_number(aggregate.get("security_stop_ci95_high"))
            and math.isclose(float(aggregate["security_stop_ci95_high"]), high, abs_tol=1e-12),
            f"tier {tier} Wilson upper bound differs in {path.name}",
        )
        rates.append(stops / expected_episodes_per_tier)
    _require(
        _is_number(report.get("worst_tier_security_stop_rate"))
        and math.isclose(float(report["worst_tier_security_stop_rate"]), min(rates), abs_tol=1e-12),
        f"worst-tier score differs from episode evidence in {path.name}",
    )
    _verify_csv_copies(path, report)
    return report


def verify_security_release(
    root: Path = ROOT,
    *,
    fingerprint_provider: Callable[[], str] = security_environment_fingerprint,
) -> dict[str, Any]:
    root = root.resolve()
    fingerprint = fingerprint_provider()
    checkpoint_sha256 = _sha256(root / SECURITY_CHECKPOINT)

    final = _verify_report(
        root / FINAL_REPORT,
        expected_fingerprint=fingerprint,
        expected_seed_start=FINAL_SEED_START,
        expected_episodes_per_tier=FINAL_EPISODES_PER_TIER,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_controller="recurrent-mappo",
    )
    actual_stops = {
        tier: sum(int(record["security_stop"]) for record in final["episodes"] if record["tier"] == tier)
        for tier in TIERS
    }
    _require(actual_stops == FINAL_STOP_COUNTS, "canonical 13M final stop counts changed")

    failed = _verify_report(
        root / FAILED_FINAL_REPORT,
        expected_fingerprint=fingerprint,
        expected_seed_start=12_000_000,
        expected_episodes_per_tier=25,
        expected_checkpoint_sha256=_read_json(root / FAILED_FINAL_REPORT).get("security_checkpoint_sha256"),
        expected_controller="recurrent-mappo",
    )
    _require(
        all(not bool(record["security_stop"]) for record in failed["episodes"]),
        "retained 12M negative evidence is no longer the zero-stop candidate",
    )
    _require(
        failed["security_checkpoint_sha256"] != checkpoint_sha256,
        "failed 12M candidate must not be the selected checkpoint",
    )

    for relative, seed_start, episodes_per_tier in (
        (VALIDATION_REPORTS[0], 11_000_000, 25),
        (VALIDATION_REPORTS[1], 11_050_000, 10),
    ):
        _verify_report(
            root / relative,
            expected_fingerprint=fingerprint,
            expected_seed_start=seed_start,
            expected_episodes_per_tier=episodes_per_tier,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_controller="recurrent-mappo",
        )

    teacher = _verify_report(
        root / TEACHER_REPORT,
        expected_fingerprint=fingerprint,
        expected_seed_start=11_000_000,
        expected_episodes_per_tier=25,
        expected_checkpoint_sha256=None,
        expected_controller="tactical-observation-only",
    )
    mean_stop_rate = fmean(float(final["tiers"][str(tier)]["security_stop_rate"]) for tier in TIERS)
    teacher_mean_stop_rate = fmean(
        float(teacher["tiers"][str(tier)]["security_stop_rate"]) for tier in TIERS
    )
    return {
        "status": "passed",
        "observation_contract": SECURITY_OBSERVATION_CONTRACT,
        "environment_fingerprint": fingerprint,
        "security_checkpoint_sha256": checkpoint_sha256,
        "runner_opponent": RUNNER_OPPONENT,
        "final_seed_start": FINAL_SEED_START,
        "final_episodes": FINAL_EPISODES_PER_TIER * len(TIERS),
        "tier_stop_rates": {
            str(tier): float(final["tiers"][str(tier)]["security_stop_rate"]) for tier in TIERS
        },
        "mean_stop_rate": mean_stop_rate,
        "teacher_mean_stop_rate": teacher_mean_stop_rate,
        "known_limitation": "tier 4 recorded zero stops in the 25-contract final slice",
    }


def main() -> None:
    try:
        report = verify_security_release()
    except SecurityReleaseEvidenceError as error:
        print(f"security release evidence failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    main()
