from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghostline.cli import build_parser
from ghostline import exporting


def test_export_cli_exposes_safe_quantization_paths() -> None:
    args = build_parser().parse_args(
        [
            "export",
            "--model",
            "checkpoint.pt",
            "--output",
            "policy.fp32.onnx",
            "--quantize",
            "--quantized-output",
            "policy.int8.onnx",
            "--deployment-output",
            "policy.deploy.onnx",
        ]
    )

    assert args.quantize is True
    assert args.quantized_output == Path("policy.int8.onnx")
    assert args.deployment_output == Path("policy.deploy.onnx")
    assert args.parity_samples == 1000


def test_export_rejects_paths_that_can_overwrite_canonical_fp32(tmp_path: Path) -> None:
    output = tmp_path / "policy.onnx"

    with pytest.raises(ValueError, match="must differ"):
        exporting._validate_paths(output, output, None)
    with pytest.raises(ValueError, match="deployment copy"):
        exporting._validate_paths(output, tmp_path / "policy.int8.onnx", output)


def test_int8_mismatch_keeps_fp32_deployment_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import onnxruntime.quantization

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    fp32 = tmp_path / "policy.onnx"
    int8 = tmp_path / "policy.int8.onnx"
    deployment = tmp_path / "policy.deploy.onnx"

    class DummyPolicy:
        recurrent = True
        recurrent_size = 256

        def cpu(self):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(exporting, "load_policy", lambda _path: DummyPolicy())
    monkeypatch.setattr(exporting, "_export_fp32", lambda _policy, _observation, path: path.write_bytes(b"verified-fp32"))
    monkeypatch.setattr(exporting, "_stamp_policy_metadata", lambda _path: {})

    parity_results = iter(
        (
            {"samples": 10, "action_mismatches": 0, "first_mismatch_index": None, "passed": True, "seed_start": 2_000_000},
            {"samples": 10, "action_mismatches": 1, "first_mismatch_index": 4, "passed": False, "seed_start": 2_000_000},
        )
    )
    monkeypatch.setattr(exporting, "_check_recurrent_action_parity", lambda *_args, **_kwargs: next(parity_results))

    def fake_quantize(*, model_input: str, model_output: str, weight_type) -> None:
        assert Path(model_input) == fp32
        assert weight_type is not None
        Path(model_output).write_bytes(b"rejected-int8")

    monkeypatch.setattr(onnxruntime.quantization, "quantize_dynamic", fake_quantize)

    report = exporting.export_policy(
        checkpoint,
        fp32,
        parity_samples=10,
        quantize=True,
        quantized_output=int8,
        deployment_output=deployment,
    )

    assert fp32.read_bytes() == b"verified-fp32"
    assert int8.read_bytes() == b"rejected-int8"
    assert deployment.read_bytes() == b"verified-fp32"
    assert report["selected_precision"] == "fp32"
    assert report["artifacts"]["dynamic_int8"]["status"] == "rejected-action-mismatch"
    assert report["deployment_copy"]["sha256"] == report["artifacts"]["fp32"]["sha256"]
    persisted = json.loads(fp32.with_suffix(".parity.json").read_text(encoding="utf-8"))
    assert persisted == report


def test_real_dynamic_int8_export_is_audited(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ghostline.model import UniversalGhostlinePolicy, save_policy

    checkpoint = tmp_path / "smoke.pt"
    fp32 = tmp_path / "smoke.onnx"
    deployment = tmp_path / "smoke.deploy.onnx"
    save_policy(UniversalGhostlinePolicy(recurrent_size=256), checkpoint, purpose="quantization-smoke")

    report = exporting.export_policy(
        checkpoint,
        fp32,
        parity_samples=1000,
        quantize=True,
        deployment_output=deployment,
    )

    assert report["artifacts"]["fp32"]["parity"]["passed"] is True
    assert report["artifacts"]["fp32"]["parity"]["samples"] == 1000
    assert report["artifacts"]["fp32"]["parity"]["sequence_horizon"] == 128
    assert report["artifacts"]["fp32"]["bytes"] > 0
    import onnx

    metadata = {item.key: item.value for item in onnx.load(fp32).metadata_props}
    assert metadata["ghostline.contract"] == "GhostlineEnv-v2"
    assert metadata["ghostline.environment_fingerprint"] == report["environment_fingerprint"]
    int8 = report["artifacts"]["dynamic_int8"]
    assert int8["status"] in {"accepted", "rejected-action-mismatch"}
    if int8["status"] == "accepted":
        int8_metadata = {item.key: item.value for item in onnx.load(int8["path"]).metadata_props}
        assert int8_metadata["ghostline.contract"] == "GhostlineEnv-v2"
        assert int8_metadata["ghostline.environment_fingerprint"] == report["environment_fingerprint"]
        assert int8["parity"]["action_mismatches"] == 0
        assert report["selected_precision"] == "dynamic-int8"
        assert report["deployment_copy"]["sha256"] == int8["sha256"]
    else:
        assert report["selected_precision"] == "fp32"
        assert report["deployment_copy"]["sha256"] == report["artifacts"]["fp32"]["sha256"]
