"""ONNX export and recurrent parity gates for the 288-action v2 runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ghostline.env_v2 import GhostlineEnvV2
from ghostline.model_v2 import (
    OBSERVATION_CONTRACT_V2,
    RunnerPolicyV2,
    load_runner_v2,
    multi_agent_environment_fingerprint,
    runner_model_fingerprint,
)
from ghostline.types_v2 import RUNNER_ACTION_COUNT_V2, ContractDirective


RUNNER_V2_ONNX_CONTRACT = "ghostline-runner-onnx-v2.0"
OBSERVATION_KEYS_V2 = (
    "ego",
    "objective",
    "directive",
    "field",
    "field_targets",
    "field_target_mask",
    "local_grid",
    "targets",
    "target_mask",
    "entities",
    "entity_mask",
    "rays",
    "action_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunnerV2OnnxPolicy(nn.Module):
    """Flatten the observation dictionary into a stable deployment ABI."""

    def __init__(self, policy: RunnerPolicyV2):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        ego: torch.Tensor,
        objective: torch.Tensor,
        directive: torch.Tensor,
        field: torch.Tensor,
        field_targets: torch.Tensor,
        field_target_mask: torch.Tensor,
        local_grid: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        rays: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation = dict(
            zip(
                OBSERVATION_KEYS_V2,
                (
                    ego,
                    objective,
                    directive,
                    field,
                    field_targets,
                    field_target_mask,
                    local_grid,
                    targets,
                    target_mask,
                    entities,
                    entity_mask,
                    rays,
                    action_mask,
                ),
                strict=True,
            )
        )
        return self.policy(observation, hidden)


def _stamp_metadata(path: Path, checkpoint_sha256: str) -> dict[str, str]:
    import onnx

    metadata = {
        "ghostline.contract": RUNNER_V2_ONNX_CONTRACT,
        "ghostline.observation_contract": OBSERVATION_CONTRACT_V2,
        "ghostline.action_count": str(RUNNER_ACTION_COUNT_V2),
        "ghostline.environment_fingerprint": multi_agent_environment_fingerprint(),
        "ghostline.model_fingerprint": runner_model_fingerprint(),
        "ghostline.checkpoint_sha256": checkpoint_sha256,
    }
    graph = onnx.load_model(str(path), load_external_data=False)
    retained = {
        item.key: item.value
        for item in graph.metadata_props
        if item.key not in metadata
    }
    del graph.metadata_props[:]
    for key, value in (*retained.items(), *metadata.items()):
        graph.metadata_props.add(key=key, value=value)
    onnx.save_model(graph, str(path))
    return metadata


def _export_graph(
    policy: RunnerPolicyV2,
    observation: Mapping[str, np.ndarray],
    output: Path,
) -> None:
    inputs = [
        torch.as_tensor(observation[key]).unsqueeze(0)
        for key in OBSERVATION_KEYS_V2
    ]
    hidden = torch.zeros(1, 1, policy.recurrent_size)
    torch.onnx.export(
        RunnerV2OnnxPolicy(policy),
        (*inputs, hidden),
        output,
        input_names=[*OBSERVATION_KEYS_V2, "hidden"],
        output_names=["logits", "value", "next_hidden"],
        dynamic_axes={
            key: {0: "batch"} for key in OBSERVATION_KEYS_V2
        }
        | {
            "hidden": {1: "batch"},
            "logits": {0: "batch"},
            "value": {0: "batch"},
            "next_hidden": {1: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )


def check_runner_v2_onnx_parity(
    policy: RunnerPolicyV2,
    onnx_path: Path,
    *,
    samples: int = 1_000,
    seed_start: int = 2_900_000,
) -> dict[str, Any]:
    """Compare deterministic recurrent actions across varied legal masks."""

    import onnxruntime as ort

    if samples <= 0:
        raise ValueError("parity samples must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=("CPUExecutionProvider",),
    )
    episode = 0
    episode_steps = 0
    tier = 1
    directive = ContractDirective.STANDARD
    env = GhostlineEnvV2(
        seed=seed_start,
        tier=tier,
        directive=directive,
    )
    observation, _ = env.reset(
        seed=seed_start,
        options={"tier": tier, "directive": directive},
    )
    torch_hidden = torch.zeros(1, 1, policy.recurrent_size)
    onnx_hidden = np.zeros((1, 1, policy.recurrent_size), dtype=np.float32)
    mismatches = 0
    first_mismatch: int | None = None
    max_logit_error = 0.0
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index in range(samples):
            feed = {
                key: np.expand_dims(observation[key], 0)
                for key in OBSERVATION_KEYS_V2
            }
            feed["hidden"] = onnx_hidden
            tensors = {
                key: torch.as_tensor(value)
                for key, value in feed.items()
                if key != "hidden"
            }
            with torch.no_grad():
                torch_logits, _, torch_hidden = policy(
                    tensors,
                    torch_hidden,
                )
            onnx_logits, _, onnx_hidden = session.run(None, feed)
            reference = int(torch.argmax(torch_logits, dim=-1).item())
            candidate = int(np.argmax(onnx_logits, axis=-1)[0])
            max_logit_error = max(
                max_logit_error,
                float(
                    np.max(
                        np.abs(
                            torch_logits.detach().cpu().numpy()
                            - onnx_logits
                        )
                    )
                ),
            )
            if reference != candidate:
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = index
            if not bool(observation["action_mask"][reference]):
                raise RuntimeError("PyTorch v2 policy selected an illegal action")
            observation, _, terminated, truncated, _ = env.step(reference)
            episode_steps += 1
            if terminated or truncated or episode_steps >= 128:
                episode += 1
                episode_steps = 0
                tier = episode % 6 + 1
                directive = ContractDirective(
                    episode % len(tuple(ContractDirective))
                )
                seed = seed_start + episode
                observation, _ = env.reset(
                    seed=seed,
                    options={
                        "tier": tier,
                        "directive": directive,
                    },
                )
                torch_hidden = torch.zeros_like(torch_hidden)
                onnx_hidden = np.zeros_like(onnx_hidden)
    finally:
        env.close()
        torch.set_num_threads(prior_threads)
    return {
        "samples": samples,
        "action_mismatches": mismatches,
        "first_mismatch_index": first_mismatch,
        "max_absolute_logit_error": max_logit_error,
        "passed": mismatches == 0,
        "seed_start": seed_start,
        "sequence_horizon": 128,
    }


def export_runner_v2_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    parity_samples: int = 1_000,
) -> dict[str, Any]:
    """Export FP32 ONNX and fail unless recurrent deterministic actions agree."""

    checkpoint = Path(checkpoint).resolve()
    output = Path(output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"v2 runner checkpoint is missing: {checkpoint}")
    policy = load_runner_v2(checkpoint, device="cpu").eval()
    env = GhostlineEnvV2(seed=2_900_000, tier=6)
    try:
        observation, _ = env.reset(seed=2_900_000, options={"tier": 6})
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    _export_graph(policy, observation, output)
    checkpoint_hash = _sha256(checkpoint)
    metadata = _stamp_metadata(output, checkpoint_hash)
    parity = check_runner_v2_onnx_parity(
        policy,
        output,
        samples=parity_samples,
    )
    report = {
        "contract": RUNNER_V2_ONNX_CONTRACT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "onnx": str(output),
        "onnx_sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "recurrent_size": policy.recurrent_size,
        "action_count": RUNNER_ACTION_COUNT_V2,
        "metadata": metadata,
        "parity": parity,
    }
    report_path = output.with_suffix(".parity.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not parity["passed"]:
        raise RuntimeError(
            f"v2 ONNX parity failed at transition {parity['first_mismatch_index']}"
        )
    return report
