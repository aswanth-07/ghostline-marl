from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ghostline.cli import build_parser
from ghostline import evaluation_v2
from ghostline.evaluation_v2 import evaluate_runner_v2_checkpoint
from ghostline.exporting_v2 import export_runner_v2_checkpoint
from ghostline.model_v2 import (
    RunnerPolicyV2,
    load_runner_v2,
    multi_agent_environment_fingerprint,
    save_runner_v2,
)


def test_v2_release_commands_are_explicit_and_do_not_replace_v1() -> None:
    train = build_parser().parse_args(
        [
            "train-runner-v2",
            "--published-v1-init",
            "models/ghostline-policy.pt",
        ]
    )
    evaluate = build_parser().parse_args(
        ["evaluate-runner-v2", "--model", "runner.pt"]
    )
    export = build_parser().parse_args(
        [
            "export-runner-v2",
            "--model",
            "runner.pt",
            "--output",
            "runner.onnx",
        ]
    )
    published = build_parser().parse_args(["train", "--dry-run"])
    assert train.command == "train-runner-v2"
    assert train.validation_interval == 100
    assert evaluate.seed_start == 20_000_000
    assert evaluate.slice_manifest == Path(
        "benchmarks/runner-v2/final-test-slices.json"
    )
    assert export.parity_samples == 1_000
    assert published.command == "train"


def test_v2_checkpoint_rejects_same_shape_model_semantic_drift(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "runner.pt"
    save_runner_v2(RunnerPolicyV2(recurrent_size=256), checkpoint)
    load_runner_v2(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_fingerprint"] = "0" * 64
    stale = tmp_path / "stale.pt"
    torch.save(payload, stale)
    with pytest.raises(RuntimeError, match="stale v2 model contract"):
        load_runner_v2(stale)


def test_tracked_v2_final_slice_is_fingerprint_bound_and_unopened() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "runner-v2"
        / "final-test-slices.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["environment_fingerprint"] == (
        multi_agent_environment_fingerprint()
    )
    reserved = manifest["slices"][0]
    assert reserved["seed_start"] == 20_000_000
    assert reserved["status"] == "reserved_unopened"
    assert reserved["episodes_per_tier"] == 500
    assert reserved["tiers"] == [1, 2, 3, 4, 5, 6]
    assert reserved["policy_kind"] == (
        "runner-v2-neural:standard,ghost,speed,greed"
    )


def test_v2_evaluator_writes_deterministic_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePolicy:
        def act(self, observation, hidden, **_kwargs):
            assert observation["action_mask"][0]
            return 0, hidden

    class FakeEnv:
        def __init__(self, *, seed, tier, directive):
            self.seed = seed
            self.tier = tier
            self.directive = directive
            self.sim = SimpleNamespace(
                elapsed_seconds=1.0,
                max_trace=0.0,
                damage_taken=0,
                detections=0,
            )

        @staticmethod
        def _observation():
            return {"action_mask": np.ones(288, dtype=np.int8)}

        def reset(self, **_kwargs):
            return self._observation(), {}

        def step(self, action):
            assert action == 0
            return self._observation(), 1.0, True, False, {
                "is_success": True,
                "fail_reason": "none",
                "duration_seconds": 1.0,
                "max_trace": 0.0,
                "damage": 0,
                "detections": 0,
                "optional_data": 0,
                "reward_total": 1.0,
                "reward_components": {"success": 1.0},
                "telemetry": {"path_efficiency": 1.0},
            }

        def close(self):
            return None

    checkpoint = tmp_path / "runner.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(evaluation_v2, "load_runner_v2", lambda *_args, **_kwargs: FakePolicy())
    monkeypatch.setattr(evaluation_v2, "GhostlineEnvV2", FakeEnv)

    def fake_init(*_args, **_kwargs):
        evaluation_v2._WORKER_POLICY = FakePolicy()
        evaluation_v2._WORKER_SECURITY_POOL = None

    monkeypatch.setattr(evaluation_v2, "_init_worker", fake_init)
    output = tmp_path / "evaluation.json"
    slice_manifest = tmp_path / "slices.json"
    fingerprint = multi_agent_environment_fingerprint()
    slice_manifest.write_text(
        json.dumps(
            {
                "manifest_contract": "ghostline-final-test-slices-v1",
                "observation_contract": "GhostlineEnv-v2",
                "environment_fingerprint": fingerprint,
                "slices": [
                    {
                        "seed_start": 20_000_000,
                        "status": "reserved_unopened",
                        "environment_fingerprint": fingerprint,
                        "policy_kind": (
                            "runner-v2-neural:standard,ghost"
                        ),
                        "episodes_per_tier": 2,
                        "tiers": [1, 2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluate_runner_v2_checkpoint(
        checkpoint,
        output,
        episodes_per_tier=2,
        tiers=(1, 2),
        directives=("standard", "ghost"),
        seed_start=20_000_000,
        workers=1,
        slice_manifest=slice_manifest,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["episodes"]) == 8
    assert report["directives"]["standard"]["1"]["success_rate"] == 1.0
    assert report["tier_worst_directive_success"] == {"1": 1.0, "2": 1.0}
    assert output.with_suffix(".csv").is_file()
    assert output.with_name("evaluation.episodes.csv").is_file()
    consumed = json.loads(
        slice_manifest.read_text(encoding="utf-8")
    )["slices"][0]
    assert consumed["status"] == "consumed"
    assert len(consumed["result"]["outputs"]) == 3
    with pytest.raises(FileExistsError):
        evaluate_runner_v2_checkpoint(
            checkpoint,
            output,
            episodes_per_tier=1,
            tiers=(1,),
            workers=1,
        )


def test_v2_onnx_export_has_recurrent_action_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "runner.pt"
    output = tmp_path / "runner.onnx"
    torch.manual_seed(71)
    save_runner_v2(RunnerPolicyV2(recurrent_size=256), checkpoint)
    report = export_runner_v2_checkpoint(
        checkpoint,
        output,
        parity_samples=16,
    )
    assert output.is_file()
    assert report["action_count"] == 288
    assert report["parity"]["samples"] == 16
    assert report["parity"]["action_mismatches"] == 0
