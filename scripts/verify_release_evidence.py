"""Fail-closed portfolio release gate for Ghostline's measured artifacts.

The evaluator, exporter, and throughput benchmark each write independently
auditable evidence.  This script binds those records to the exact checkpoint,
deployment ONNX graph, frozen environment source, and one-way final-test slice
present in the release checkout.  It never runs or reopens final-test episodes.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, median
import sys
from typing import Any, Callable

from ghostline.onnx_contract import OnnxPolicyContract, environment_fingerprint, validate_onnx_policy


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_CONTRACT = "GhostlineEnv-v2"
FINAL_REPORT_CONTRACT = "ghostline-final-evaluation-v2"
SLICE_MANIFEST_CONTRACT = "ghostline-final-test-slices-v1"
THROUGHPUT_REPORT_CONTRACT = "ghostline-headless-throughput-v1"
FINAL_SEED_START = 7_000_000
EPISODES_PER_TIER = 500
TIERS = tuple(range(1, 7))
ACCEPTANCE = {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.85}
MIN_PARITY_SAMPLES = 1_000
MIN_PARITY_HORIZON = 128
MIN_THROUGHPUT_DECISIONS_PER_SECOND = 5_000.0
MIN_BENCHMARK_DECISIONS_PER_WORKER = 10_000
MIN_BENCHMARK_WORKERS = 4
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

CHECKPOINT = Path("models/ghostline-policy.pt")
DEPLOYMENT_ONNX = Path("models/ghostline-policy.onnx")
FINAL_REPORT = Path("benchmarks/neural/champion-final-7m-500.json")
PARITY_REPORT = Path("benchmarks/neural/champion-onnx-parity.json")
SLICE_MANIFEST = Path("benchmarks/final-test-slices.json")
THROUGHPUT_REPORT = Path("benchmarks/system/headless-throughput.json")
DEMO_VIDEO = Path("videos/ghostline-demo.mp4")


class ReleaseEvidenceError(RuntimeError):
    """Raised when tracked evidence cannot authorize a portfolio release."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseEvidenceError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReleaseEvidenceError(f"required release evidence is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseEvidenceError(f"could not read release evidence: {path}") from error
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"release evidence root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        source = path.open("rb")
    except OSError as error:
        raise ReleaseEvidenceError(f"required release artifact is missing: {path}") from error
    digest = hashlib.sha256()
    with source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _expected_seed(tier: int, episode_index: int) -> int:
    return FINAL_SEED_START + tier * 100_000 + episode_index


def _verify_episode_records(report: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    records = report.get("episode_records")
    _require(isinstance(records, list), "final report has no episode_records list")
    _require(
        len(records) == EPISODES_PER_TIER * len(TIERS),
        "final report must contain exactly 3,000 episode records",
    )
    grouped: dict[int, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    continuous_metrics = (
        "duration_seconds",
        "max_trace",
        "policy_latency_total_ms",
        "reward_total",
    )
    integer_metrics = (
        "damage",
        "damage_by_guard",
        "damage_by_drone",
        "detections",
        "optional_data",
        "pulse_uses",
        "decision_count",
    )
    for expected_ordinal, row in enumerate(records):
        _require(isinstance(row, dict), f"episode record {expected_ordinal} is not an object")
        expected_tier = TIERS[expected_ordinal // EPISODES_PER_TIER]
        expected_index = expected_ordinal % EPISODES_PER_TIER
        _require(row.get("tier") == expected_tier, "episode records are not in canonical tier order")
        _require(row.get("episode_index") == expected_index, "episode indexes are incomplete or reordered")
        _require(
            row.get("seed") == _expected_seed(expected_tier, expected_index),
            "episode seed does not match the final-test seed formula",
        )
        _require(isinstance(row.get("success"), bool), "episode success must be boolean")
        _require(isinstance(row.get("failure_reason"), str), "episode failure reason is missing")
        _require(_is_sha256(row.get("action_sha256")), "episode action hash is missing or malformed")
        for name in continuous_metrics:
            _require(_is_number(row.get(name)), f"episode metric {name!r} is missing or non-finite")
        for name in integer_metrics:
            _require(
                isinstance(row.get(name), int)
                and not isinstance(row.get(name), bool)
                and int(row[name]) >= 0,
                f"episode metric {name!r} must be a non-negative integer",
            )
        _require(int(row["decision_count"]) > 0, "episode decision count must be positive")
        _require(float(row["duration_seconds"]) > 0.0, "episode duration must be positive")
        _require(0.0 <= float(row["max_trace"]) <= 100.0, "episode max trace is outside 0..100")
        _require(float(row["policy_latency_total_ms"]) >= 0.0, "episode latency cannot be negative")
        _require(
            int(row["damage"]) == int(row["damage_by_guard"]) + int(row["damage_by_drone"]),
            "episode damage attribution does not sum to total damage",
        )
        path_efficiency = row.get("path_efficiency")
        _require(
            path_efficiency is None
            or (_is_number(path_efficiency) and 0.0 <= float(path_efficiency) <= 1.0),
            "episode path efficiency must be null or lie in 0..1",
        )
        median_latency = row.get("median_policy_latency_ms")
        _require(
            median_latency is None
            or (_is_number(median_latency) and float(median_latency) >= 0.0),
            "episode median policy latency must be null or non-negative",
        )
        components = row.get("reward_components")
        _require(isinstance(components, dict) and components, "episode reward accounting is missing")
        _require(
            all(isinstance(key, str) and _is_number(value) for key, value in components.items()),
            "episode reward accounting contains invalid components",
        )
        _require(
            math.isclose(
                float(row["reward_total"]),
                sum(float(value) for value in components.values()),
                abs_tol=1e-9,
            ),
            "episode reward components do not sum to reward_total",
        )
        grouped[expected_tier].append(row)
    return grouped


def _verify_final_report(
    path: Path,
    *,
    fingerprint: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], str]:
    report = _read_json(path)
    _require(report.get("report_contract") == FINAL_REPORT_CONTRACT, "unsupported final report contract")
    _require(report.get("observation_contract") == OBSERVATION_CONTRACT, "final report is not Env-v2")
    _require(report.get("environment_fingerprint") == fingerprint, "final report fingerprint is stale")
    _require(report.get("checkpoint_sha256") == checkpoint_sha256, "final report checkpoint hash does not match")
    _require(report.get("release_audit") is True, "final report was not produced by a release audit")
    _require(report.get("deterministic_actions") is True, "final report did not use deterministic actions")
    _require(report.get("policy_kind") == "neural", "final report is not a neural-policy evaluation")
    _require(report.get("seed_start") == FINAL_SEED_START, "final report did not consume the reserved 7M slice")
    _require(report.get("episodes_per_tier") == EPISODES_PER_TIER, "final report must use 500 episodes per tier")
    _require(report.get("tiers_evaluated") == list(TIERS), "final report must cover tiers 1-6")
    _require(report.get("meets_acceptance_thresholds") is True, "neural policy misses acceptance thresholds")
    _require(report.get("acceptance_thresholds") == {str(k): v for k, v in ACCEPTANCE.items()}, "final thresholds were changed")
    audit_id = report.get("slice_audit_id")
    _require(_is_sha256(audit_id), "final report slice audit identity is malformed")
    grouped = _verify_episode_records(report)

    tiers = report.get("tiers")
    _require(isinstance(tiers, dict) and set(tiers) == {str(tier) for tier in TIERS}, "tier aggregates are incomplete")
    for tier, rows in grouped.items():
        aggregate = tiers[str(tier)]
        _require(isinstance(aggregate, dict), f"tier {tier} aggregate is not an object")
        successes = sum(int(row["success"]) for row in rows)
        success_rate = successes / EPISODES_PER_TIER
        _require(aggregate.get("episodes") == EPISODES_PER_TIER, f"tier {tier} episode total is incorrect")
        _require(aggregate.get("successes") == successes, f"tier {tier} success total is incorrect")
        _require(
            _is_number(aggregate.get("success_rate"))
            and math.isclose(float(aggregate["success_rate"]), success_rate, abs_tol=1e-12),
            f"tier {tier} aggregate success rate does not match episode evidence",
        )
        _require(success_rate >= ACCEPTANCE[tier], f"tier {tier} misses its neural success threshold")
        interval = aggregate.get("wilson_95")
        expected_interval = _wilson(successes, EPISODES_PER_TIER)
        _require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(_is_number(value) for value in interval)
            and all(math.isclose(float(value), expected, abs_tol=1e-12) for value, expected in zip(interval, expected_interval)),
            f"tier {tier} Wilson interval does not match the episode evidence",
        )
        expected_failures = Counter(
            str(row["failure_reason"]) for row in rows if not bool(row["success"])
        )
        _require(
            aggregate.get("failure_reasons") == {key: expected_failures[key] for key in sorted(expected_failures)},
            f"tier {tier} failure taxonomy does not match episode evidence",
        )
        expected_metrics = {
            "median_duration_seconds": float(median(float(row["duration_seconds"]) for row in rows)),
            "median_max_trace": float(median(float(row["max_trace"]) for row in rows)),
            "mean_damage": float(fmean(float(row["damage"]) for row in rows)),
            "mean_guard_damage": float(fmean(float(row["damage_by_guard"]) for row in rows)),
            "mean_drone_damage": float(fmean(float(row["damage_by_drone"]) for row in rows)),
            "mean_detections": float(fmean(float(row["detections"]) for row in rows)),
            "optional_data_rate": float(
                fmean(float(int(row["optional_data"]) > 0) for row in rows)
            ),
        }
        path_values = [float(row["path_efficiency"]) for row in rows if row["path_efficiency"] is not None]
        latency_values = [
            float(row["median_policy_latency_ms"])
            for row in rows
            if row["median_policy_latency_ms"] is not None
        ]
        decision_count = sum(int(row["decision_count"]) for row in rows)
        latency_total = sum(float(row["policy_latency_total_ms"]) for row in rows)
        expected_optional = {
            "mean_path_efficiency": float(fmean(path_values)) if path_values else None,
            "mean_policy_latency_ms": latency_total / decision_count if decision_count else None,
            "median_episode_policy_latency_ms": float(median(latency_values)) if latency_values else None,
        }
        for metric, expected in expected_metrics.items():
            _require(
                _is_number(aggregate.get(metric))
                and math.isclose(float(aggregate[metric]), expected, abs_tol=1e-12),
                f"tier {tier} aggregate metric {metric!r} does not match episode evidence",
            )
        for metric, expected in expected_optional.items():
            actual = aggregate.get(metric)
            _require(
                (actual is None and expected is None)
                or (
                    _is_number(actual)
                    and expected is not None
                    and math.isclose(float(actual), expected, abs_tol=1e-12)
                ),
                f"tier {tier} aggregate metric {metric!r} does not match episode evidence",
            )
    return report, str(audit_id)


def _verify_slice_manifest(
    path: Path,
    *,
    root: Path,
    fingerprint: str,
    audit_id: str,
    report: dict[str, Any],
) -> None:
    manifest = _read_json(path)
    _require(manifest.get("manifest_contract") == SLICE_MANIFEST_CONTRACT, "unsupported final-slice manifest")
    _require(manifest.get("observation_contract") == OBSERVATION_CONTRACT, "slice manifest is not Env-v2")
    _require(manifest.get("environment_fingerprint") == fingerprint, "slice manifest fingerprint is stale")
    _require(not path.with_name(f"{path.name}.lock").exists(), "final-test slice remains locked")
    slices = manifest.get("slices")
    _require(isinstance(slices, list), "slice manifest has no slices list")
    matches = [item for item in slices if isinstance(item, dict) and item.get("seed_start") == FINAL_SEED_START]
    _require(len(matches) == 1, "slice manifest must contain exactly one 7M record")
    selected = matches[0]
    _require(selected.get("status") == "consumed", "reserved 7M final-test slice has not been consumed")
    _require(selected.get("environment_fingerprint") == fingerprint, "7M slice fingerprint is stale")
    _require(selected.get("policy_kind") == "neural", "7M slice was not reserved for a neural policy")
    _require(selected.get("episodes_per_tier") == EPISODES_PER_TIER, "7M slice episode count changed")
    _require(selected.get("tiers") == list(TIERS), "7M slice tier set changed")
    result = selected.get("result")
    _require(isinstance(result, dict), "consumed 7M slice has no result record")
    _require(result.get("audit_id") == audit_id, "slice and final report audit identities differ")
    _require(result.get("meets_acceptance_thresholds") is True, "slice result did not pass acceptance")
    outputs = result.get("outputs")
    _require(isinstance(outputs, list), "slice result does not hash its outputs")
    expected_relatives = (
        FINAL_REPORT,
        FINAL_REPORT.with_suffix(".csv"),
        FINAL_REPORT.with_suffix(".episodes.csv"),
    )
    records = {
        item.get("path"): item
        for item in outputs
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    _require(set(records) == {path.as_posix() for path in expected_relatives}, "slice output set is not canonical")
    for relative in expected_relatives:
        record = records[relative.as_posix()]
        artifact = root / relative
        _require(artifact.is_file(), f"slice output is missing: {relative.as_posix()}")
        _require(record.get("bytes") == artifact.stat().st_size, f"slice output size changed: {relative.as_posix()}")
        _require(record.get("sha256") == _sha256(artifact), f"slice output hash changed: {relative.as_posix()}")
    _require(report.get("slice_manifest") == SLICE_MANIFEST.as_posix(), "final report names a noncanonical slice manifest")


def _parity_record(record: object, *, label: str) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{label} parity artifact is missing")
    parity = record.get("parity")
    _require(isinstance(parity, dict), f"{label} recurrent parity record is missing")
    _require(parity.get("passed") is True, f"{label} recurrent parity did not pass")
    _require(int(parity.get("samples", 0)) >= MIN_PARITY_SAMPLES, f"{label} parity used fewer than 1,000 transitions")
    _require(int(parity.get("action_mismatches", -1)) == 0, f"{label} deterministic actions differ from PyTorch")
    _require(int(parity.get("sequence_horizon", 0)) >= MIN_PARITY_HORIZON, f"{label} parity horizon is too short")
    _require(sorted({int(value) for value in parity.get("tiers", [])}) == list(TIERS), f"{label} parity does not cover all tiers")
    _require(_is_sha256(record.get("sha256")), f"{label} ONNX hash is malformed")
    return record


def _verify_parity_report(
    path: Path,
    *,
    checkpoint_sha256: str,
    onnx_sha256: str,
    onnx_bytes: int,
    fingerprint: str,
) -> dict[str, Any]:
    report = _read_json(path)
    _require(report.get("report_version") == 2, "unsupported ONNX parity report")
    _require(report.get("status") == "passed", "ONNX export parity did not pass")
    _require(report.get("observation_contract") == OBSERVATION_CONTRACT, "parity report is not Env-v2")
    _require(report.get("environment_fingerprint") == fingerprint, "parity report fingerprint is stale")
    _require(report.get("checkpoint_sha256") == checkpoint_sha256, "parity checkpoint does not match the champion")
    _require(int(report.get("parity_samples", 0)) >= MIN_PARITY_SAMPLES, "top-level parity sample count is too small")
    _require(int(report.get("mismatches", -1)) == 0, "top-level parity summary has action mismatches")
    artifacts = report.get("artifacts")
    _require(isinstance(artifacts, dict), "parity report has no artifact records")
    fp32 = _parity_record(artifacts.get("fp32"), label="FP32")
    selected_precision = report.get("selected_precision")
    _require(selected_precision in ("fp32", "dynamic-int8"), "parity report selected an unsupported precision")
    selected_key = "fp32" if selected_precision == "fp32" else "dynamic_int8"
    selected = _parity_record(artifacts.get(selected_key), label=str(selected_precision))
    if selected_key == "dynamic_int8":
        _require(selected.get("status") == "accepted", "INT8 was deployed without accepted parity")
    _require(selected.get("sha256") == onnx_sha256, "deployed ONNX bytes are not the selected parity artifact")
    deployment = report.get("deployment_copy")
    _require(isinstance(deployment, dict), "parity report has no deployment-copy record")
    _require(deployment.get("sha256") == onnx_sha256, "deployment-copy hash does not match tracked ONNX")
    _require(deployment.get("bytes") == onnx_bytes, "deployment-copy size does not match tracked ONNX")
    _require(fp32.get("parity", {}).get("action_mismatches") == 0, "FP32 is not an exact deterministic fallback")
    return report


def _verify_throughput(path: Path, *, fingerprint: str) -> dict[str, Any]:
    report = _read_json(path)
    _require(report.get("report_contract") == THROUGHPUT_REPORT_CONTRACT, "unsupported throughput report")
    _require(report.get("observation_contract") == OBSERVATION_CONTRACT, "throughput report is not Env-v2")
    _require(report.get("environment_fingerprint") == fingerprint, "throughput report fingerprint is stale")
    _require(report.get("tier") == 6, "release throughput must benchmark tier 6")
    _require(report.get("action_repeat_ticks") == 6, "throughput report action-repeat contract changed")
    decisions = report.get("decisions_per_worker")
    workers = report.get("workers")
    total = report.get("total_decisions")
    _require(isinstance(decisions, int) and decisions >= MIN_BENCHMARK_DECISIONS_PER_WORKER, "release benchmark is too short")
    _require(isinstance(workers, int) and workers >= MIN_BENCHMARK_WORKERS, "release benchmark uses too few workers")
    _require(total == decisions * workers, "throughput total decision count is inconsistent")
    wall = report.get("wall_elapsed_seconds")
    aggregate = report.get("aggregate_decisions_per_second")
    ticks = report.get("aggregate_simulation_ticks_per_second")
    _require(_is_number(wall) and float(wall) > 0, "throughput wall time is invalid")
    _require(_is_number(aggregate), "aggregate decision throughput is missing")
    _require(
        math.isclose(float(aggregate), total / float(wall), rel_tol=1e-9),
        "aggregate decision throughput does not match raw timing",
    )
    _require(float(aggregate) >= MIN_THROUGHPUT_DECISIONS_PER_SECOND, "headless simulator misses 5,000 decisions/s")
    _require(_is_number(ticks) and math.isclose(float(ticks), float(aggregate) * 6, rel_tol=1e-9), "simulation tick throughput is inconsistent")
    _require(report.get("meets_minimum") is True, "throughput report did not pass its configured gate")
    _require(
        _is_number(report.get("minimum_decisions_per_second"))
        and float(report["minimum_decisions_per_second"]) >= MIN_THROUGHPUT_DECISIONS_PER_SECOND,
        "throughput run did not request the 5,000 decisions/s release minimum",
    )
    elapsed = report.get("worker_elapsed_seconds")
    _require(isinstance(elapsed, list) and len(elapsed) == workers and all(_is_number(value) and float(value) > 0 for value in elapsed), "worker timings are incomplete")
    system = report.get("system")
    _require(isinstance(system, dict), "throughput system provenance is missing")
    logical_cpus = system.get("logical_cpu_count")
    _require(isinstance(logical_cpus, int) and logical_cpus >= workers, "throughput worker count exceeds recorded CPU capacity")
    return report


def _verify_demo_video(path: Path) -> str:
    _require(path.is_file(), f"required demo video is missing: {path}")
    _require(path.stat().st_size >= 100_000, "demo video is too small to be a portfolio recording")
    try:
        with path.open("rb") as source:
            signature = source.read(32)
    except OSError as error:
        raise ReleaseEvidenceError(f"could not read demo video: {path}") from error
    _require(b"ftyp" in signature, "demo recording is not an MP4 file")
    return _sha256(path)


def verify_release(
    root: Path = ROOT,
    *,
    onnx_validator: Callable[..., OnnxPolicyContract] = validate_onnx_policy,
) -> dict[str, Any]:
    """Verify all canonical release artifacts without mutating them."""

    root = root.resolve()
    fingerprint = environment_fingerprint(root / "src" / "ghostline")
    checkpoint_path = root / CHECKPOINT
    onnx_path = root / DEPLOYMENT_ONNX
    checkpoint_sha256 = _sha256(checkpoint_path)
    onnx_sha256 = _sha256(onnx_path)
    report, audit_id = _verify_final_report(
        root / FINAL_REPORT,
        fingerprint=fingerprint,
        checkpoint_sha256=checkpoint_sha256,
    )
    _verify_slice_manifest(
        root / SLICE_MANIFEST,
        root=root,
        fingerprint=fingerprint,
        audit_id=audit_id,
        report=report,
    )
    parity = _verify_parity_report(
        root / PARITY_REPORT,
        checkpoint_sha256=checkpoint_sha256,
        onnx_sha256=onnx_sha256,
        onnx_bytes=onnx_path.stat().st_size,
        fingerprint=fingerprint,
    )
    contract = onnx_validator(onnx_path, expected_fingerprint=fingerprint)
    _require(contract.recurrent_size in (256, 384), "release ONNX uses a non-final recurrent width")
    _require(
        parity.get("recurrent_size") == contract.recurrent_size,
        "parity report recurrent width does not match the selected ONNX graph",
    )
    throughput = _verify_throughput(root / THROUGHPUT_REPORT, fingerprint=fingerprint)
    demo_sha256 = _verify_demo_video(root / DEMO_VIDEO)
    return {
        "status": "passed",
        "observation_contract": OBSERVATION_CONTRACT,
        "environment_fingerprint": fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_sha256": onnx_sha256,
        "final_report_sha256": _sha256(root / FINAL_REPORT),
        "slice_manifest_sha256": _sha256(root / SLICE_MANIFEST),
        "parity_report_sha256": _sha256(root / PARITY_REPORT),
        "throughput_report_sha256": _sha256(root / THROUGHPUT_REPORT),
        "onnx_precision": parity["selected_precision"],
        "recurrent_size": contract.recurrent_size,
        "final_seed_start": FINAL_SEED_START,
        "episodes": EPISODES_PER_TIER * len(TIERS),
        "tier_success_rates": {
            str(tier): report["tiers"][str(tier)]["success_rate"] for tier in TIERS
        },
        "aggregate_decisions_per_second": throughput["aggregate_decisions_per_second"],
        "demo_video_sha256": demo_sha256,
    }


def main() -> None:
    try:
        summary = verify_release()
    except (ReleaseEvidenceError, RuntimeError, ValueError, TypeError) as error:
        print(f"release evidence failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    main()
