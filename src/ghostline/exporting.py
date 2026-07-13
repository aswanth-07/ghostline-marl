from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ghostline.env import GhostlineEnv
from ghostline.model import UniversalGhostlinePolicy, load_policy


OBS_KEYS = ("ego", "objective", "local_grid", "targets", "target_mask", "entities", "entity_mask", "rays", "action_mask")
PARITY_SEED_START = 2_000_000
PARITY_SEQUENCE_HORIZON = 128


class OnnxPolicy(nn.Module):
    def __init__(self, policy: UniversalGhostlinePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, ego, objective, local_grid, targets, target_mask, entities, entity_mask, rays, action_mask, hidden):
        obs = dict(zip(OBS_KEYS, (ego, objective, local_grid, targets, target_mask, entities, entity_mask, rays, action_mask)))
        logits, value, next_hidden = self.policy(obs, hidden)
        return logits, value, next_hidden


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _default_quantized_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.int8{output.suffix}")


def _environment_fingerprint() -> str:
    from ghostline.imitation import training_environment_fingerprint

    return training_environment_fingerprint()


def _validate_paths(output: Path, quantized_output: Path | None, deployment_output: Path | None) -> None:
    output_resolved = output.resolve()
    if quantized_output is not None and quantized_output.resolve() == output_resolved:
        raise ValueError("The INT8 candidate path must differ from the canonical FP32 output")
    if deployment_output is not None:
        deployment_resolved = deployment_output.resolve()
        reserved = {output_resolved}
        if quantized_output is not None:
            reserved.add(quantized_output.resolve())
        if deployment_resolved in reserved:
            raise ValueError("The deployment copy must differ from the audited FP32 and INT8 artifact paths")


def _export_fp32(policy: UniversalGhostlinePolicy, observation: dict[str, np.ndarray], output: Path) -> None:
    tensors = [torch.as_tensor(observation[key]).unsqueeze(0) for key in OBS_KEYS]
    hidden = torch.zeros(1, 1, policy.recurrent_size)
    torch.onnx.export(
        OnnxPolicy(policy),
        (*tensors, hidden),
        output,
        input_names=[*OBS_KEYS, "hidden"],
        output_names=["logits", "value", "next_hidden"],
        dynamic_axes={key: {0: "batch"} for key in OBS_KEYS}
        | {
            "hidden": {1: "batch"},
            "logits": {0: "batch"},
            "value": {0: "batch"},
            "next_hidden": {1: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    _stamp_policy_metadata(output)


def _stamp_policy_metadata(path: Path) -> dict[str, str]:
    """Bind an ONNX graph to the exact frozen player/environment contract."""

    import onnx

    metadata = {
        "ghostline.contract": "GhostlineEnv-v2",
        "ghostline.environment_fingerprint": _environment_fingerprint(),
    }
    model = onnx.load_model(str(path), load_external_data=False)
    retained = [(item.key, item.value) for item in model.metadata_props if item.key not in metadata]
    del model.metadata_props[:]
    for key, value in (*retained, *metadata.items()):
        model.metadata_props.add(key=key, value=value)
    onnx.save_model(model, str(path))
    return metadata


def _check_recurrent_action_parity(
    policy: UniversalGhostlinePolicy,
    onnx_path: Path,
    *,
    parity_samples: int,
    seed_start: int = PARITY_SEED_START,
) -> dict[str, Any]:
    """Replay a deterministic PyTorch trajectory through one recurrent ONNX graph."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    episode_index = 0
    episode_steps = 0
    env = GhostlineEnv(seed=seed_start, tier=1)
    observation, _ = env.reset(seed=seed_start, options={"tier": 1})
    torch_hidden = torch.zeros(1, 1, policy.recurrent_size)
    onnx_hidden = np.zeros((1, 1, policy.recurrent_size), dtype=np.float32)
    mismatches = 0
    first_mismatch: int | None = None
    checked = 0
    previous_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        while checked < parity_samples:
            feed = {key: np.expand_dims(observation[key], 0) for key in OBS_KEYS}
            feed["hidden"] = onnx_hidden
            with torch.no_grad():
                torch_inputs = {key: torch.as_tensor(feed[key]) for key in OBS_KEYS}
                logits, _, torch_hidden = policy(torch_inputs, torch_hidden)
            onnx_logits, _, onnx_hidden = session.run(None, feed)
            reference_action = int(torch.argmax(logits, dim=-1).item())
            candidate_action = int(np.argmax(onnx_logits, axis=-1)[0])
            if candidate_action != reference_action:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = checked
            observation, _, terminated, truncated, _ = env.step(reference_action)
            checked += 1
            episode_steps += 1
            if terminated or truncated or episode_steps >= PARITY_SEQUENCE_HORIZON:
                episode_index += 1
                episode_steps = 0
                tier = episode_index % 6 + 1
                observation, _ = env.reset(seed=seed_start + episode_index, options={"tier": tier})
                torch_hidden = torch.zeros_like(torch_hidden)
                onnx_hidden = np.zeros_like(onnx_hidden)
    finally:
        env.close()
        torch.set_num_threads(previous_torch_threads)
    return {
        "samples": checked,
        "action_mismatches": mismatches,
        "first_mismatch_index": first_mismatch,
        "passed": mismatches == 0,
        "seed_start": seed_start,
        "sequence_horizon": PARITY_SEQUENCE_HORIZON,
        "tiers": [1, 2, 3, 4, 5, 6],
    }


def _copy_selected(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return _artifact(destination)


def export_policy(
    checkpoint: Path,
    output: Path,
    *,
    parity_samples: int = 1000,
    quantize: bool = False,
    quantized_output: Path | None = None,
    deployment_output: Path | None = None,
) -> dict[str, Any]:
    """Export canonical FP32 ONNX and optionally gate a dynamic-INT8 candidate.

    ``output`` is always the canonical FP32 graph. Quantization never overwrites it.
    If ``deployment_output`` is supplied, it receives the INT8 graph only when that
    graph has zero recurrent deterministic-action mismatches; otherwise it receives
    the verified FP32 fallback.
    """
    if parity_samples <= 0:
        raise ValueError("parity_samples must be positive")
    if quantized_output is not None and not quantize:
        raise ValueError("quantized_output requires quantize=True")
    if quantize and quantized_output is None:
        quantized_output = _default_quantized_path(output)
    _validate_paths(output, quantized_output, deployment_output)

    policy = load_policy(checkpoint).cpu().eval()
    if not policy.recurrent:
        raise ValueError("The web/player ONNX contract requires a recurrent policy checkpoint")

    env = GhostlineEnv(seed=PARITY_SEED_START, tier=6)
    try:
        observation, _ = env.reset(seed=PARITY_SEED_START)
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    _export_fp32(policy, observation, output)

    fp32_parity = _check_recurrent_action_parity(policy, output, parity_samples=parity_samples)
    fp32_artifact = _artifact(output) | {"precision": "fp32", "parity": fp32_parity}
    report: dict[str, Any] = {
        "report_version": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx": str(output),
        "parity_samples": fp32_parity["samples"],
        "mismatches": fp32_parity["action_mismatches"],
        "recurrent_size": policy.recurrent_size,
        "observation_contract": "GhostlineEnv-v2",
        "environment_fingerprint": _environment_fingerprint(),
        "quantization": "dynamic-int8" if quantize else None,
        "artifacts": {"fp32": fp32_artifact},
        "selected_precision": "fp32",
        "selected_path": str(output),
    }
    report_path = output.with_suffix(".parity.json")

    if not fp32_parity["passed"]:
        report["status"] = "fp32-parity-failed"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"FP32 ONNX parity failed: {fp32_parity['action_mismatches']}/{fp32_parity['samples']} "
            "deterministic actions differ"
        )

    selected_source = output
    if quantize:
        assert quantized_output is not None
        quantized_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=str(output),
                model_output=str(quantized_output),
                weight_type=QuantType.QInt8,
            )
            # Quantizers are not required to preserve arbitrary metadata.
            # Re-stamp before parity so the exact artifact being audited is
            # also the one accepted by the fail-closed web release gate.
            _stamp_policy_metadata(quantized_output)
            int8_parity = _check_recurrent_action_parity(
                policy,
                quantized_output,
                parity_samples=parity_samples,
            )
            int8_artifact = _artifact(quantized_output) | {
                "precision": "dynamic-int8",
                "parity": int8_parity,
                "size_reduction_fraction": 1.0 - (quantized_output.stat().st_size / output.stat().st_size),
                "status": "accepted" if int8_parity["passed"] else "rejected-action-mismatch",
            }
            report["artifacts"]["dynamic_int8"] = int8_artifact
            if int8_parity["passed"]:
                selected_source = quantized_output
                report["selected_precision"] = "dynamic-int8"
                report["selected_path"] = str(quantized_output)
        except Exception as exc:  # FP32 remains a verified release-safe fallback.
            report["artifacts"]["dynamic_int8"] = {
                "path": str(quantized_output),
                "precision": "dynamic-int8",
                "status": "quantization-error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if deployment_output is not None:
        deployment_artifact = _copy_selected(selected_source, deployment_output)
        deployment_artifact["precision"] = report["selected_precision"]
        report["deployment_copy"] = deployment_artifact
    report["status"] = "passed"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
