"""Application-owned operations, with no model deserialization or uploaded code."""

import hashlib
import platform
import time
from collections.abc import Mapping

import numpy as np
import torch

from neurolabs import __version__
from neurolabs.schema import Dense, ModelSpec


def validate_parameters(spec: ModelSpec, parameters: Mapping[str, torch.Tensor]) -> None:
    if set(parameters) != {tensor.id for tensor in spec.tensors}:
        raise ValueError("Parameter names do not match the model")
    for tensor in spec.tensors:
        value = parameters[tensor.id]
        if value.dtype != torch.float32 or tuple(value.shape) != tensor.shape:
            raise ValueError(f"Shape or dtype mismatch for {tensor.id}")
        if value.device.type != "cpu" or not torch.isfinite(value).all().item():
            raise ValueError("Parameters must be finite CPU float32 tensors")


def execute(
    spec: ModelSpec,
    parameters: Mapping[str, torch.Tensor],
    inputs: list[list[float]] | tuple[tuple[float, ...], ...],
    revision_id: str,
    captures: tuple[str, ...] = (),
) -> dict:
    validate_parameters(spec, parameters)
    if not 1 <= len(inputs) <= 256 or any(len(row) != spec.input_width for row in inputs):
        raise ValueError(f"Expected 1–256 input rows of width {spec.input_width}")
    valid_captures = {op.output for op in spec.operations} | {spec.input_name}
    if not set(captures) <= valid_captures:
        raise ValueError("Unknown activation capture")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(inputs, dtype="<f4")
    if not np.isfinite(array).all():
        raise ValueError("Inputs must remain finite when converted to float32")
    input_hash = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
    x = torch.from_numpy(array.copy())
    activation_values: dict[str, list] = {}
    captured_bytes = 0
    started = time.perf_counter()
    if spec.input_name in captures:
        activation_values[spec.input_name] = x.tolist()
        captured_bytes += x.numel() * x.element_size()
    with torch.inference_mode():
        for operation in spec.operations:
            if isinstance(operation, Dense):
                bias = parameters[operation.bias] if operation.bias else None
                x = torch.nn.functional.linear(x, parameters[operation.weight], bias)
            elif operation.kind == "softmax":
                x = torch.softmax(x, dim=-1)
            elif operation.kind != "identity":
                x = getattr(torch, operation.kind)(x)
            if not torch.isfinite(x).all().item():
                raise ValueError(
                    f"Nonfinite computation at {operation.id}; reduce input/weight size"
                )
            if operation.output in captures:
                captured_bytes += x.numel() * x.element_size()
                if captured_bytes > 16 * 1024 * 1024:
                    raise ValueError("Requested captures exceed the 16 MiB activation limit")
                activation_values[operation.output] = x.tolist()
    return {
        "revision_id": revision_id,
        "input_hash": input_hash,
        "model_spec_hash": hashlib.sha256(spec.model_dump_json().encode()).hexdigest(),
        "inputs": array.tolist(),
        "outputs": x.tolist(),
        "activations": activation_values,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "provenance": {
            "application": __version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
