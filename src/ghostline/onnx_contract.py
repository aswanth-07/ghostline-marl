"""Build-time validation for Ghostline's browser/desktop ONNX contract.

This module deliberately has no PyTorch or runtime-inference imports.  Release
builders can therefore inspect a selected policy before it enters either the
static web bundle or the player executable without pulling training code into
the shipped application.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


ENVIRONMENT_FINGERPRINT_FILES = (
    "config.py",
    "env.py",
    "generation.py",
    "policies.py",
    "simulation.py",
    "types.py",
)
POLICY_INPUT_SHAPES: dict[str, list[int]] = {
    "ego": [1, 24],
    "objective": [1, 8],
    "local_grid": [1, 8, 15, 15],
    "targets": [1, 5, 10],
    "target_mask": [1, 5],
    "entities": [1, 12, 13],
    "entity_mask": [1, 12],
    "rays": [1, 24, 3],
    "action_mask": [1, 36],
}
POLICY_FLOAT_INPUTS = frozenset(
    {"ego", "objective", "local_grid", "targets", "entities", "rays", "hidden"}
)
POLICY_MASK_INPUTS = frozenset({"target_mask", "entity_mask", "action_mask"})


@dataclass(frozen=True)
class OnnxPolicyContract:
    recurrent_size: int
    input_shapes: dict[str, list[int]]
    metadata: dict[str, str]


def environment_fingerprint(package: Path | None = None) -> str:
    """Hash the exact source set used by neural training and export gates."""

    package = package or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ENVIRONMENT_FINGERPRINT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def validate_onnx_policy(
    model: Path,
    *,
    expected_fingerprint: str,
) -> OnnxPolicyContract:
    """Fail closed unless ``model`` is the complete player-equivalent v2 graph."""

    try:
        import onnx
    except ImportError as error:  # pragma: no cover - release extras own this dependency
        raise RuntimeError(
            "ONNX policy inspection requires the Ghostline [build] or [web] extra"
        ) from error

    try:
        graph = onnx.load_model(str(model), load_external_data=False)
    except Exception as error:
        raise RuntimeError(f"could not read selected ONNX policy metadata: {model}") from error
    external_initializers = [
        tensor.name
        for tensor in graph.graph.initializer
        if tensor.data_location == int(onnx.TensorProto.EXTERNAL) or tensor.external_data
    ]
    if external_initializers:
        raise RuntimeError(
            "ONNX release policy must be a single self-contained file; external tensors: "
            + ", ".join(external_initializers[:5])
        )

    def static_shape(value, *, recurrent: bool = False) -> list[int]:
        tensor_type = value.type.tensor_type
        shape: list[int] = []
        for index, dimension in enumerate(tensor_type.shape.dim):
            # Exported batch dimensions may be symbolic. Browser and packaged
            # player inference both run one environment per recurrent state.
            numeric = int(dimension.dim_value)
            if numeric > 0:
                shape.append(numeric)
            elif index == 0 or (recurrent and index == 1):
                shape.append(1)
            else:
                raise RuntimeError(
                    f"ONNX tensor {value.name!r} has a non-static feature dimension"
                )
        return shape

    inputs = {
        value.name: static_shape(value, recurrent=value.name == "hidden")
        for value in graph.graph.input
    }
    input_types = {
        value.name: int(value.type.tensor_type.elem_type) for value in graph.graph.input
    }
    required = {*POLICY_INPUT_SHAPES, "hidden"}
    missing = sorted(required - inputs.keys())
    if missing:
        raise RuntimeError(f"ONNX policy is missing release inputs: {', '.join(missing)}")
    for name, expected in POLICY_INPUT_SHAPES.items():
        if inputs[name] != expected:
            raise RuntimeError(f"ONNX input {name!r} has shape {inputs[name]}; expected {expected}")
    hidden = inputs["hidden"]
    if len(hidden) != 3 or hidden[:2] != [1, 1] or hidden[-1] <= 0:
        raise RuntimeError(f"ONNX recurrent input has unsupported shape: {hidden}")

    expected_types = {
        **{name: int(onnx.TensorProto.FLOAT) for name in POLICY_FLOAT_INPUTS},
        **{name: int(onnx.TensorProto.INT8) for name in POLICY_MASK_INPUTS},
    }
    for name, expected in expected_types.items():
        if input_types[name] != expected:
            raise RuntimeError(
                f"ONNX input {name!r} has tensor type {input_types[name]}; expected {expected}"
            )

    outputs = {
        value.name: static_shape(value, recurrent=value.name == "next_hidden")
        for value in graph.graph.output
    }
    output_types = {
        value.name: int(value.type.tensor_type.elem_type) for value in graph.graph.output
    }
    expected_outputs = {
        "logits": [1, 36],
        "value": [1],
        "next_hidden": [1, 1, hidden[-1]],
    }
    missing_outputs = sorted(expected_outputs.keys() - outputs.keys())
    if missing_outputs:
        raise RuntimeError(
            f"ONNX policy is missing release outputs: {', '.join(missing_outputs)}"
        )
    for name, expected in expected_outputs.items():
        if outputs[name] != expected:
            raise RuntimeError(f"ONNX output {name!r} has shape {outputs[name]}; expected {expected}")
        if output_types[name] != int(onnx.TensorProto.FLOAT):
            raise RuntimeError(f"ONNX output {name!r} must use float32 tensors")

    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    if metadata.get("ghostline.contract") != "GhostlineEnv-v2":
        raise RuntimeError("ONNX policy metadata does not declare GhostlineEnv-v2")
    if metadata.get("ghostline.environment_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            "ONNX policy was not exported from the current frozen environment fingerprint"
        )
    return OnnxPolicyContract(
        recurrent_size=hidden[-1],
        input_shapes={name: inputs[name] for name in (*POLICY_INPUT_SHAPES, "hidden")},
        metadata=metadata,
    )
