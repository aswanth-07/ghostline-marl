from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostline import evaluation
from ghostline.cli import build_parser
from ghostline.env import GhostlineEnv
from ghostline.evaluation import (
    DEFAULT_RELEASE_SEED_START,
    DEFAULT_SLICE_MANIFEST,
    FINAL_SLICE_MANIFEST_CONTRACT,
    _action_sequence_hash,
    _aggregate_csv,
    _build_report,
    _episode_record,
    _episodes_csv,
    _open_final_slice,
    _stable_json,
)
from ghostline.model import current_environment_fingerprint
from ghostline.seeds import final_test_seed


ROOT = Path(__file__).resolve().parents[1]


def _manifest(
    path: Path,
    *,
    fingerprint: str = "fingerprint",
    status: str = "reserved_unopened",
    episodes: int = 2,
    tiers: tuple[int, ...] = (1, 2),
) -> Path:
    path.write_text(
        json.dumps(
            {
                "manifest_contract": FINAL_SLICE_MANIFEST_CONTRACT,
                "observation_contract": "GhostlineEnv-v2",
                "environment_fingerprint": fingerprint,
                "slices": [
                    {
                        "seed_start": DEFAULT_RELEASE_SEED_START,
                        "status": status,
                        "environment_fingerprint": fingerprint,
                        "policy_kind": "neural",
                        "episodes_per_tier": episodes,
                        "tiers": list(tiers),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _info(*, success: bool) -> dict[str, object]:
    return {
        "is_success": success,
        "fail_reason": "none" if success else "contract_expired",
        "duration_seconds": 12.5,
        "max_trace": 44.0,
        "damage": 1,
        "damage_by_guard": 1,
        "damage_by_drone": 0,
        "detections": 2,
        "optional_data": 1,
        "pulse_uses": 1,
        "telemetry": {"path_efficiency": 0.75},
        "reward_components": {"success": 20.0, "time": -0.25},
    }


def test_cli_defaults_to_the_tracked_unopened_7m_slice() -> None:
    args = build_parser().parse_args(["evaluate", "--model", "models/ghostline-policy.pt"])

    assert args.seed_start == 7_000_000
    assert args.slice_manifest == Path("benchmarks/final-test-slices.json")
    assert args.output == Path("benchmarks/neural/champion-final-7m-500.json")

    manifest = json.loads((ROOT / DEFAULT_SLICE_MANIFEST).read_text(encoding="utf-8"))
    reserved = next(item for item in manifest["slices"] if item["seed_start"] == 7_000_000)
    assert manifest["environment_fingerprint"] == current_environment_fingerprint()
    assert reserved["environment_fingerprint"] == current_environment_fingerprint()
    assert reserved["status"] == "reserved_unopened"
    assert reserved["episodes_per_tier"] == 500
    assert reserved["tiers"] == [1, 2, 3, 4, 5, 6]


def test_slice_lease_is_one_way_and_hashes_all_outputs(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "slices.json")
    output = tmp_path / "report.json"
    lease = _open_final_slice(
        manifest_path=manifest_path,
        seed_start=DEFAULT_RELEASE_SEED_START,
        episodes=2,
        tiers=(1, 2),
        environment_fingerprint="fingerprint",
        policy_kind="neural",
        checkpoint_sha256="checkpoint",
        output=output,
    )

    opened = json.loads(manifest_path.read_text(encoding="utf-8"))["slices"][0]
    assert opened["status"] == "opened_locked"
    assert opened["active_audit"]["audit_id"] == lease.audit_id
    assert lease.lock_path.is_file()

    outputs = (output, output.with_suffix(".csv"), output.with_suffix(".episodes.csv"))
    for index, path in enumerate(outputs):
        path.write_text(f"evidence-{index}\n", encoding="utf-8")
    lease.finalize({"meets_acceptance_thresholds": True}, outputs)

    consumed = json.loads(manifest_path.read_text(encoding="utf-8"))["slices"][0]
    assert consumed["status"] == "consumed"
    assert consumed["result"]["audit_id"] == lease.audit_id
    assert all(len(item["sha256"]) == 64 for item in consumed["result"]["outputs"])
    assert not lease.lock_path.exists()

    with pytest.raises(RuntimeError, match="only reserved_unopened"):
        _open_final_slice(
            manifest_path=manifest_path,
            seed_start=DEFAULT_RELEASE_SEED_START,
            episodes=2,
            tiers=(1, 2),
            environment_fingerprint="fingerprint",
            policy_kind="neural",
            checkpoint_sha256="checkpoint",
            output=tmp_path / "second.json",
        )
    assert not lease.lock_path.exists()


def test_failed_attempt_retires_the_slice_without_a_reopen_path(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path / "slices.json")
    lease = _open_final_slice(
        manifest_path=manifest_path,
        seed_start=DEFAULT_RELEASE_SEED_START,
        episodes=2,
        tiers=(1, 2),
        environment_fingerprint="fingerprint",
        policy_kind="neural",
        checkpoint_sha256="checkpoint",
        output=tmp_path / "report.json",
    )
    lease.abort(RuntimeError("simulated worker failure"))

    retired = json.loads(manifest_path.read_text(encoding="utf-8"))["slices"][0]
    assert retired["status"] == "aborted_retired"
    assert retired["result"]["error_type"] == "RuntimeError"
    assert "simulated worker failure" not in manifest_path.read_text(encoding="utf-8")
    assert not lease.lock_path.exists()


@pytest.mark.parametrize(
    ("status", "fingerprint", "match"),
    (
        ("consumed", "fingerprint", "only reserved_unopened"),
        ("reserved_unopened", "stale", "fingerprint"),
    ),
)
def test_manifest_preflight_rejects_reused_or_stale_slices_without_opening(
    tmp_path: Path, status: str, fingerprint: str, match: str
) -> None:
    manifest_path = _manifest(tmp_path / "slices.json", status=status, fingerprint=fingerprint)
    before = manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match=match):
        _open_final_slice(
            manifest_path=manifest_path,
            seed_start=DEFAULT_RELEASE_SEED_START,
            episodes=2,
            tiers=(1, 2),
            environment_fingerprint="fingerprint",
            policy_kind="neural",
            checkpoint_sha256="checkpoint",
            output=tmp_path / "report.json",
        )
    assert manifest_path.read_bytes() == before
    assert not manifest_path.with_name(f"{manifest_path.name}.lock").exists()


def test_report_and_csv_bind_exact_seed_order_and_action_hashes() -> None:
    records = [
        _episode_record(
            _info(success=index == 0),
            tier=1,
            seed=final_test_seed(DEFAULT_RELEASE_SEED_START, 1, index),
            episode_index=index,
            actions=[0, 8 + index, 35],
            policy_latencies_ms=[1.0, 2.0, 3.0],
        )
        for index in range(2)
    ]
    report = _build_report(
        model=Path("models/champion.pt"),
        checkpoint_sha256="a" * 64,
        baseline="teacher",
        seed_start=DEFAULT_RELEASE_SEED_START,
        episodes=2,
        tiers=(1,),
        environment_fingerprint="b" * 64,
        audit_id="c" * 64,
        slice_manifest=Path("benchmarks/final-test-slices.json"),
        episode_records=list(reversed(records)),
    )

    assert report["checkpoint_sha256"] == "a" * 64
    assert report["environment_fingerprint"] == "b" * 64
    assert [row["seed"] for row in report["episode_records"]] == [
        final_test_seed(DEFAULT_RELEASE_SEED_START, 1, 0),
        final_test_seed(DEFAULT_RELEASE_SEED_START, 1, 1),
    ]
    assert report["episode_records"][0]["action_sha256"] == _action_sequence_hash([0, 8, 35])
    assert _stable_json(report) == _stable_json(report)
    assert _aggregate_csv(report) == _aggregate_csv(report)
    assert _episodes_csv(report) == _episodes_csv(report)
    assert "action_sha256" in _episodes_csv(report).splitlines()[0]
    assert "reward_total" in _episodes_csv(report).splitlines()[0]

    broken = [dict(records[0]), dict(records[1])]
    broken[1]["seed"] += 1
    with pytest.raises(RuntimeError, match="scheduled seed order"):
        _build_report(
            model=Path("models/champion.pt"),
            checkpoint_sha256="a" * 64,
            baseline="teacher",
            seed_start=DEFAULT_RELEASE_SEED_START,
            episodes=2,
            tiers=(1,),
            environment_fingerprint="b" * 64,
            audit_id="c" * 64,
            slice_manifest=Path("benchmarks/final-test-slices.json"),
            episode_records=broken,
        )


def test_real_terminal_info_reconstructs_exact_reward_accounting() -> None:
    env = GhostlineEnv(seed=10101, tier=1)
    env.reset(seed=10101, options={"training_lesson": 1})
    env.sim.elapsed_ticks = int(env.sim.level.mission_seconds * 60) - 1
    try:
        _, _, terminated, truncated, info = env.step(0)
    finally:
        env.close()

    assert not terminated and truncated
    expected = {
        key.removeprefix("reward_"): float(value)
        for key, value in info.items()
        if key.startswith("reward_") and key != "reward_total"
    }
    record = _episode_record(
        info,
        tier=1,
        seed=10101,
        episode_index=0,
        actions=[0],
    )
    assert record["reward_components"] == expected
    assert record["reward_total"] == pytest.approx(sum(expected.values()))
    assert record["reward_total"] == pytest.approx(info["reward_total"])

    tampered = dict(info)
    tampered["reward_total"] = float(info["reward_total"]) + 1.0
    with pytest.raises(RuntimeError, match="do not sum"):
        _episode_record(
            tampered,
            tier=1,
            seed=10101,
            episode_index=0,
            actions=[0],
        )


def test_public_evaluator_consumes_manifest_and_writes_all_evidence_without_real_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _manifest(
        tmp_path / "slices.json",
        episodes=1,
        tiers=(1,),
    )
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"current-fingerprint-checkpoint")
    output = tmp_path / "champion.json"

    class InlinePool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, jobs):
            return [function(job) for job in jobs]

    def fake_model_episode(job: tuple[int, int, int]) -> dict[str, object]:
        tier, seed, episode_index = job
        return _episode_record(
            _info(success=True),
            tier=tier,
            seed=seed,
            episode_index=episode_index,
            actions=[0, 17, 35],
            policy_latencies_ms=[1.0, 1.5, 2.0],
        )

    monkeypatch.setattr(evaluation, "current_environment_fingerprint", lambda: "fingerprint")
    monkeypatch.setattr(evaluation, "load_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(evaluation, "_model_episode", fake_model_episode)

    report = evaluation.evaluate(
        model=checkpoint,
        episodes=1,
        tier=1,
        output=output,
        workers=1,
        seed_start=DEFAULT_RELEASE_SEED_START,
        slice_manifest=manifest_path,
    )

    assert report["release_audit"] is True
    assert report["episode_records"][0]["seed"] == final_test_seed(
        DEFAULT_RELEASE_SEED_START, 1, 0
    )
    assert output.is_file()
    assert output.with_suffix(".csv").is_file()
    assert output.with_suffix(".episodes.csv").is_file()
    consumed = json.loads(manifest_path.read_text(encoding="utf-8"))["slices"][0]
    assert consumed["status"] == "consumed"
    assert not manifest_path.with_name(f"{manifest_path.name}.lock").exists()
