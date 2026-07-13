from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from ghostline.seeds import FINAL_TEST_SEED_START, VALIDATION_SEED_END, validation_seed
from scripts import evaluate_validation_policy as evaluator


def _fake_result(job: tuple[int, int, int, int]) -> dict[str, object]:
    tier, ordinal, validation_index, seed = job
    success = tier not in (2, 6)
    decision_count = 2
    mean_latency = tier / 10.0
    return {
        "tier": tier,
        "episode_ordinal": ordinal,
        "validation_index": validation_index,
        "seed": seed,
        "success": success,
        "failure_reason": "none" if success else ("timer" if tier == 2 else "integrity_lost"),
        "damage": int(not success),
        "detections": tier,
        "duration_seconds": float(10 + tier),
        "path_efficiency": 0.50 + tier / 100.0,
        "decision_count": decision_count,
        "policy_latency_total_ms": mean_latency * decision_count,
        "mean_policy_latency_ms": mean_latency,
        "median_policy_latency_ms": mean_latency,
        "p95_policy_latency_ms": mean_latency,
        "action_sha256": hashlib.sha256(bytes((tier,))).hexdigest(),
    }


def test_validation_jobs_are_disjoint_bounded_and_deterministically_ordered() -> None:
    jobs = evaluator.build_validation_jobs(episodes=2, validation_offset=7_998)

    assert len(jobs) == 12
    assert jobs[0] == (1, 0, 7_998, validation_seed(1, 7_998))
    assert jobs[1] == (1, 1, 7_999, validation_seed(1, 7_999))
    assert jobs[-1] == (6, 1, 7_999, validation_seed(6, 7_999))
    seeds = [job[3] for job in jobs]
    assert len(seeds) == len(set(seeds))
    assert max(seeds) <= VALIDATION_SEED_END
    assert all(seed < FINAL_TEST_SEED_START for seed in seeds)

    with pytest.raises(ValueError, match="leaves its per-tier block"):
        evaluator.build_validation_jobs(episodes=3, validation_offset=7_998)
    with pytest.raises(ValueError, match="overlap"):
        evaluator.build_validation_jobs(episodes=1, validation_offset=0, tiers=(1, 1, 6))
    with pytest.raises(ValueError, match="1..6"):
        evaluator.build_validation_jobs(episodes=1, validation_offset=0, tiers=(0, 6))


def test_worker_initializer_forces_one_cpu_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    marker = object()
    monkeypatch.setattr(evaluator.torch, "set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr(
        evaluator.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )
    monkeypatch.setattr(
        evaluator.torch,
        "use_deterministic_algorithms",
        lambda value: calls.append(("deterministic", value)),
    )

    def fake_load(path: Path, *, device: str):
        calls.append(("load", (path, device)))
        return marker

    monkeypatch.setattr(evaluator, "load_policy", fake_load)
    evaluator._WORKER_POLICY = None
    evaluator._init_policy_worker("candidate.pt")

    assert evaluator._WORKER_POLICY is marker
    assert ("intra", 1) in calls
    assert ("interop", 1) in calls
    assert ("deterministic", True) in calls
    assert ("load", (Path("candidate.pt"), "cpu")) in calls
    evaluator._WORKER_POLICY = None


def test_mock_closed_loop_run_writes_auditable_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"current neural checkpoint")
    output = tmp_path / "validation.json"
    load_calls: list[tuple[Path, str]] = []
    observed_jobs: list[tuple[int, int, int, int]] = []

    monkeypatch.setattr(evaluator, "current_environment_fingerprint", lambda: "f" * 64)

    def fake_load(path: Path, *, device: str):
        load_calls.append((path, device))
        return object()

    def fake_run(checkpoint_path, jobs, *, workers):
        assert checkpoint_path == checkpoint
        assert workers == 3
        observed_jobs.extend(jobs)
        # Deliberately reverse worker completion order; the report must sort it.
        return [_fake_result(job) for job in reversed(jobs)]

    monkeypatch.setattr(evaluator, "load_policy", fake_load)
    monkeypatch.setattr(evaluator, "_run_jobs", fake_run)

    report = evaluator.evaluate_validation_policy(
        checkpoint=checkpoint,
        episodes=1,
        validation_offset=123,
        output=output,
        workers=3,
    )

    assert load_calls == [(checkpoint, "cpu")]
    assert [job[0] for job in observed_jobs] == [1, 2, 3, 4, 5, 6]
    assert report["environment_fingerprint"] == "f" * 64
    assert report["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["final_test_seeds_used"] is False
    assert report["worst_tier"] == 2
    assert report["worst_tier_selection_tuple"][:2] == [0.0, 0.0]
    assert report["selection_tuple_fields"][0] == "worst_tier_success_rate"
    assert report["meets_single_window_thresholds"] is False
    assert report["consecutive_windows_required_for_acceptance"] == 2
    assert report["release_eligible"] is False
    assert report["tiers"]["2"]["failure_reasons"] == {"timer": 1}
    assert report["tiers"]["6"]["failure_reasons"] == {"integrity_lost": 1}
    assert [
        report["tiers"][str(tier)]["episode_records"][0]["seed"]
        for tier in range(1, 7)
    ] == [validation_seed(tier, 123) for tier in range(1, 7)]

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["checkpoint_sha256"] == report["checkpoint_sha256"]
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [int(row["tier"]) for row in csv_rows] == [1, 2, 3, 4, 5, 6]
    assert json.loads(csv_rows[1]["failure_reasons_json"]) == {"timer": 1}
    assert all(row["environment_fingerprint"] == "f" * 64 for row in csv_rows)


def test_stale_checkpoint_gate_runs_before_workers_or_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "stale.pt"
    checkpoint.write_bytes(b"stale")
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(evaluator, "current_environment_fingerprint", lambda: "current")

    def reject_stale(_path: Path, *, device: str):
        assert device == "cpu"
        raise RuntimeError("stale environment fingerprint")

    monkeypatch.setattr(evaluator, "load_policy", reject_stale)
    monkeypatch.setattr(
        evaluator,
        "_run_jobs",
        lambda *_args, **_kwargs: pytest.fail("workers must not start for a stale checkpoint"),
    )

    with pytest.raises(RuntimeError, match="stale environment fingerprint"):
        evaluator.evaluate_validation_policy(
            checkpoint=checkpoint,
            episodes=1,
            validation_offset=0,
            output=output,
            workers=1,
        )
    assert not output.exists()
    assert not output.with_suffix(".csv").exists()
