from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghostline.onnx_contract import ENVIRONMENT_FINGERPRINT_FILES, environment_fingerprint
from scripts.benchmark_ghostline import _build_report
from scripts import verify_release_evidence as release_evidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tier_aggregate(successes: int) -> dict[str, object]:
    return {
        "episodes": release_evidence.EPISODES_PER_TIER,
        "successes": successes,
        "success_rate": successes / release_evidence.EPISODES_PER_TIER,
        "wilson_95": list(
            release_evidence._wilson(successes, release_evidence.EPISODES_PER_TIER)
        ),
        "median_duration_seconds": 42.0,
        "median_max_trace": 20.0,
        "mean_damage": 0.0,
        "mean_guard_damage": 0.0,
        "mean_drone_damage": 0.0,
        "mean_detections": 0.0,
        "optional_data_rate": 1.0,
        "mean_path_efficiency": 0.82,
        "mean_policy_latency_ms": 1.1,
        "median_episode_policy_latency_ms": 1.0,
        "failure_reasons": {},
    }


def _episode(tier: int, index: int) -> dict[str, object]:
    return {
        "tier": tier,
        "episode_index": index,
        "seed": release_evidence.FINAL_SEED_START + tier * 100_000 + index,
        "success": True,
        "failure_reason": "none",
        "duration_seconds": 42.0,
        "max_trace": 20.0,
        "damage": 0,
        "damage_by_guard": 0,
        "damage_by_drone": 0,
        "detections": 0,
        "optional_data": 1,
        "pulse_uses": 1,
        "path_efficiency": 0.82,
        "decision_count": 70,
        "action_sha256": hashlib.sha256(f"{tier}:{index}".encode()).hexdigest(),
        "policy_latency_total_ms": 77.0,
        "median_policy_latency_ms": 1.0,
        "reward_total": 19.8,
        "reward_components": {"success": 20.0, "time": -0.2},
    }


def _refresh_slice_output_hashes(root: Path) -> None:
    manifest_path = root / release_evidence.SLICE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = next(
        item for item in manifest["slices"] if item["seed_start"] == release_evidence.FINAL_SEED_START
    )["result"]
    for record in result["outputs"]:
        path = root / record["path"]
        record["bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    _write_json(manifest_path, manifest)


def _write_valid_release_tree(root: Path) -> dict[str, object]:
    package = root / "src" / "ghostline"
    package.mkdir(parents=True)
    for index, name in enumerate(ENVIRONMENT_FINGERPRINT_FILES):
        (package / name).write_text(f"# frozen source {index}\n", encoding="utf-8")
    fingerprint = environment_fingerprint(package)

    checkpoint = root / release_evidence.CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-neural-checkpoint")
    checkpoint_sha256 = _sha256(checkpoint)
    onnx = root / release_evidence.DEPLOYMENT_ONNX
    onnx.write_bytes(b"frozen-onnx-policy")
    onnx_sha256 = _sha256(onnx)

    audit_id = "c" * 64
    records = [
        _episode(tier, index)
        for tier in release_evidence.TIERS
        for index in range(release_evidence.EPISODES_PER_TIER)
    ]
    report = {
        "report_contract": release_evidence.FINAL_REPORT_CONTRACT,
        "observation_contract": release_evidence.OBSERVATION_CONTRACT,
        "environment_fingerprint": fingerprint,
        "deterministic_actions": True,
        "policy_kind": "neural",
        "policy": release_evidence.CHECKPOINT.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "release_audit": True,
        "seed_namespace": "final_test_7_000_000_plus",
        "seed_start": release_evidence.FINAL_SEED_START,
        "seed_formula": "seed_start + tier * 100000 + episode_index",
        "episodes_per_tier": release_evidence.EPISODES_PER_TIER,
        "tiers_evaluated": list(release_evidence.TIERS),
        "acceptance_thresholds": {
            str(tier): threshold for tier, threshold in release_evidence.ACCEPTANCE.items()
        },
        "meets_acceptance_thresholds": True,
        "slice_manifest": release_evidence.SLICE_MANIFEST.as_posix(),
        "slice_audit_id": audit_id,
        "tiers": {
            str(tier): _tier_aggregate(release_evidence.EPISODES_PER_TIER)
            for tier in release_evidence.TIERS
        },
        "episode_records": records,
    }
    final_report = root / release_evidence.FINAL_REPORT
    _write_json(final_report, report)
    final_report.with_suffix(".csv").write_text("aggregate\n", encoding="utf-8")
    final_report.with_suffix(".episodes.csv").write_text("episodes\n", encoding="utf-8")

    outputs = []
    for relative in (
        release_evidence.FINAL_REPORT,
        release_evidence.FINAL_REPORT.with_suffix(".csv"),
        release_evidence.FINAL_REPORT.with_suffix(".episodes.csv"),
    ):
        path = root / relative
        outputs.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        root / release_evidence.SLICE_MANIFEST,
        {
            "manifest_contract": release_evidence.SLICE_MANIFEST_CONTRACT,
            "observation_contract": release_evidence.OBSERVATION_CONTRACT,
            "environment_fingerprint": fingerprint,
            "slices": [
                {
                    "seed_start": release_evidence.FINAL_SEED_START,
                    "status": "consumed",
                    "environment_fingerprint": fingerprint,
                    "policy_kind": "neural",
                    "episodes_per_tier": release_evidence.EPISODES_PER_TIER,
                    "tiers": list(release_evidence.TIERS),
                    "result": {
                        "audit_id": audit_id,
                        "meets_acceptance_thresholds": True,
                        "outputs": outputs,
                    },
                }
            ],
        },
    )

    parity = {
        "report_version": 2,
        "status": "passed",
        "checkpoint_sha256": checkpoint_sha256,
        "parity_samples": release_evidence.MIN_PARITY_SAMPLES,
        "mismatches": 0,
        "observation_contract": release_evidence.OBSERVATION_CONTRACT,
        "environment_fingerprint": fingerprint,
        "recurrent_size": 384,
        "selected_precision": "fp32",
        "artifacts": {
            "fp32": {
                "path": "models/ghostline-policy.fp32.onnx",
                "bytes": onnx.stat().st_size,
                "sha256": onnx_sha256,
                "precision": "fp32",
                "parity": {
                    "samples": release_evidence.MIN_PARITY_SAMPLES,
                    "action_mismatches": 0,
                    "passed": True,
                    "sequence_horizon": release_evidence.MIN_PARITY_HORIZON,
                    "tiers": list(release_evidence.TIERS),
                },
            }
        },
        "deployment_copy": {
            "path": release_evidence.DEPLOYMENT_ONNX.as_posix(),
            "bytes": onnx.stat().st_size,
            "sha256": onnx_sha256,
        },
    }
    _write_json(root / release_evidence.PARITY_REPORT, parity)

    worker_elapsed = [6.0, 6.2, 6.1, 6.3]
    decisions = release_evidence.MIN_BENCHMARK_DECISIONS_PER_WORKER
    workers = len(worker_elapsed)
    total = decisions * workers
    wall = 6.5
    aggregate = total / wall
    _write_json(
        root / release_evidence.THROUGHPUT_REPORT,
        {
            "report_contract": release_evidence.THROUGHPUT_REPORT_CONTRACT,
            "observation_contract": release_evidence.OBSERVATION_CONTRACT,
            "environment_fingerprint": fingerprint,
            "tier": 6,
            "action_repeat_ticks": 6,
            "decisions_per_worker": decisions,
            "workers": workers,
            "total_decisions": total,
            "wall_elapsed_seconds": wall,
            "worker_elapsed_seconds": worker_elapsed,
            "aggregate_decisions_per_second": aggregate,
            "aggregate_simulation_ticks_per_second": aggregate * 6,
            "median_worker_decisions_per_second": 1_626.0,
            "resets": 2,
            "minimum_decisions_per_second": release_evidence.MIN_THROUGHPUT_DECISIONS_PER_SECOND,
            "meets_minimum": True,
            "system": {"logical_cpu_count": 24, "python": "3.13.5"},
        },
    )
    video = root / release_evidence.DEMO_VIDEO
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100_000)
    return report


def _validator(path: Path, *, expected_fingerprint: str):
    assert path.name == release_evidence.DEPLOYMENT_ONNX.name
    assert len(expected_fingerprint) == 64
    return SimpleNamespace(recurrent_size=384)


def test_release_evidence_binds_all_selected_artifacts(tmp_path: Path) -> None:
    _write_valid_release_tree(tmp_path)

    summary = release_evidence.verify_release(tmp_path, onnx_validator=_validator)

    assert summary["status"] == "passed"
    assert summary["episodes"] == 3_000
    assert summary["recurrent_size"] == 384
    assert summary["tier_success_rates"] == {str(tier): 1.0 for tier in range(1, 7)}


def test_release_evidence_rejects_a_policy_below_any_tier_gate(tmp_path: Path) -> None:
    report = _write_valid_release_tree(tmp_path)
    tier_six = [row for row in report["episode_records"] if row["tier"] == 6]
    for row in tier_six[:76]:
        row["success"] = False
        row["failure_reason"] = "contract_expired"
    successes = release_evidence.EPISODES_PER_TIER - 76
    report["tiers"]["6"] = _tier_aggregate(successes) | {
        "failure_reasons": {"contract_expired": 76}
    }
    _write_json(tmp_path / release_evidence.FINAL_REPORT, report)
    _refresh_slice_output_hashes(tmp_path)

    with pytest.raises(release_evidence.ReleaseEvidenceError, match="tier 6 misses"):
        release_evidence.verify_release(tmp_path, onnx_validator=_validator)


def test_release_evidence_rejects_changed_deployment_bytes(tmp_path: Path) -> None:
    _write_valid_release_tree(tmp_path)
    with (tmp_path / release_evidence.DEPLOYMENT_ONNX).open("ab") as output:
        output.write(b"tampered")

    with pytest.raises(release_evidence.ReleaseEvidenceError, match="selected parity artifact"):
        release_evidence.verify_release(tmp_path, onnx_validator=_validator)


def test_headless_benchmark_report_exposes_raw_provenance() -> None:
    report = _build_report(
        decisions=10_000,
        tier=6,
        workers=4,
        wall_elapsed=4.0,
        results=((3.8, 1), (3.9, 2), (4.0, 3), (4.1, 4)),
        minimum_decisions_per_second=5_000.0,
        fingerprint="f" * 64,
    )

    assert report["report_contract"] == "ghostline-headless-throughput-v1"
    assert report["total_decisions"] == 40_000
    assert report["aggregate_decisions_per_second"] == 10_000.0
    assert report["aggregate_simulation_ticks_per_second"] == 60_000.0
    assert report["meets_minimum"] is True
    assert report["worker_elapsed_seconds"] == [3.8, 3.9, 4.0, 4.1]


def test_environment_fingerprint_is_checkout_line_ending_invariant(tmp_path: Path) -> None:
    lf_package = tmp_path / "lf" / "ghostline"
    crlf_package = tmp_path / "crlf" / "ghostline"
    lf_package.mkdir(parents=True)
    crlf_package.mkdir(parents=True)
    for index, name in enumerate(ENVIRONMENT_FINGERPRINT_FILES):
        lines = f"# source {index}\nVALUE = {index}\n".encode()
        (lf_package / name).write_bytes(lines)
        (crlf_package / name).write_bytes(lines.replace(b"\n", b"\r\n"))

    assert environment_fingerprint(lf_package) == environment_fingerprint(crlf_package)

    changed = crlf_package / ENVIRONMENT_FINGERPRINT_FILES[0]
    changed.write_bytes(changed.read_bytes().replace(b"VALUE = 0", b"VALUE = 99"))
    assert environment_fingerprint(lf_package) != environment_fingerprint(crlf_package)
