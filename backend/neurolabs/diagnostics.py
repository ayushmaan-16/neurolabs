"""Batch-scoped activation audits and behavior comparisons of immutable snapshots."""

import hashlib
from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch

from neurolabs.runtime import execute
from neurolabs.schema import Activation, Dense, ModelSpec, TensorSpec


def parameter_hash(parameters: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(parameters.items()):
        digest.update(key.encode() + b"\0" + str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.detach().contiguous().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def analyze(
    spec: ModelSpec,
    parameters: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
    inputs,
    revision_id: str,
    baseline_revision_id: str,
) -> dict:
    """Compare exact same float32 inputs; statistics describe this batch only."""
    if not 2 <= len(inputs) <= 256:
        raise ValueError("An audit requires 2–256 input rows")
    stages = []
    index = 0
    while index < len(spec.operations):
        operation = spec.operations[index]
        following = spec.operations[index + 1] if index + 1 < len(spec.operations) else None
        paired = isinstance(operation, Dense) and isinstance(following, Activation)
        activation = (
            following if paired else operation if isinstance(operation, Activation) else None
        )
        stages.append(
            (
                operation.id,
                activation.output if activation else operation.output,
                activation.kind if activation else "linear",
            )
        )
        index += 2 if paired else 1
    # Capture all operations, including uncommon consecutive activations, so the
    # final-output comparison always uses the runtime's actual final output.
    captures = tuple(operation.output for operation in spec.operations)
    current = execute(spec, parameters, inputs, revision_id, captures)
    reference = execute(spec, baseline, inputs, baseline_revision_id, captures)
    layers = []
    for stage_id, output, kind in stages:
        values = np.asarray(current["activations"][output], dtype=np.float64)
        original = np.asarray(reference["activations"][output], dtype=np.float64)
        delta = values - original
        zero = (values == 0).mean(axis=0)
        original_zero = (original == 0).mean(axis=0)
        saturated = None
        if kind == "tanh":
            saturated = (np.abs(values) >= 0.99).mean(axis=0)
        elif kind == "sigmoid":
            saturated = ((values <= 0.01) | (values >= 0.99)).mean(axis=0)
        inactive = (zero == 1) if kind == "relu" else np.zeros(values.shape[1], dtype=bool)
        neurons = [
            {
                "index": index,
                "mean": float(values[:, index].mean()),
                "std": float(values[:, index].std()),
                "min": float(values[:, index].min()),
                "max": float(values[:, index].max()),
                "zero_fraction": float(zero[index]),
                "baseline_zero_fraction": float(original_zero[index]),
                "inactive_on_batch": bool(inactive[index]),
                "saturation_fraction": float(saturated[index]) if saturated is not None else None,
                "delta_rms": float(np.sqrt(np.mean(delta[:, index] ** 2))),
            }
            for index in range(values.shape[1])
        ]
        layers.append(
            {
                "id": stage_id,
                "activation": kind,
                "width": values.shape[1],
                "rms": float(np.sqrt(np.mean(values**2))),
                "baseline_rms": float(np.sqrt(np.mean(original**2))),
                "delta_rms": float(np.sqrt(np.mean(delta**2))),
                "inactive_neurons": int(inactive.sum()),
                "neurons": neurons,
            }
        )
    outputs = np.asarray(current["outputs"], dtype=np.float64)
    original_outputs = np.asarray(reference["outputs"], dtype=np.float64)
    changes = np.abs(outputs - original_outputs)
    classification = spec.task == "classification" and outputs.shape[1] > 1
    flips = (outputs.argmax(axis=1) != original_outputs.argmax(axis=1)) if classification else None
    return {
        "format_version": 1,
        "model_id": spec.id,
        "model_spec_hash": current["model_spec_hash"],
        "revision_id": revision_id,
        "baseline_revision_id": baseline_revision_id,
        "input_hash": current["input_hash"],
        "parameter_hash": parameter_hash(parameters),
        "baseline_parameter_hash": parameter_hash(baseline),
        "sample_count": len(inputs),
        "thresholds": {"tanh_abs_min": 0.99, "sigmoid_low": 0.01, "sigmoid_high": 0.99},
        "scope": (
            "Observed behavior on the supplied batch; "
            "not proof of global inactivity or model quality."
        ),
        "summary": {
            "inactive_relu_neurons": sum(layer["inactive_neurons"] for layer in layers),
            "mean_abs_output_change": float(changes.mean()),
            "max_abs_output_change": float(changes.max()),
            "argmax_flips": int(flips.sum()) if flips is not None else None,
        },
        "layers": layers,
        "cases": [
            {
                "index": index,
                "inputs": current["inputs"][index],
                "current_output": current["outputs"][index],
                "baseline_output": reference["outputs"][index],
                "max_abs_change": float(changes[index].max()),
                "argmax_changed": bool(flips[index]) if flips is not None else None,
            }
            for index in range(len(inputs))
        ],
        "provenance": {"current": current["provenance"], "baseline": reference["provenance"]},
    }


def sequential_snapshot(model: torch.nn.Sequential):
    """Copy supported, exact built-in modules without invoking their forward methods.

    This Python-only bridge accepts an in-memory model owned by the caller. It is
    not a model-file loader, does not unpickle anything, and never changes the model.
    """
    if type(model) is not torch.nn.Sequential:
        raise ValueError("Expected an exact torch.nn.Sequential")
    if "forward" in model.__dict__ or model._forward_hooks or model._forward_pre_hooks:
        raise ValueError("Forward overrides and hooks are not supported")
    modules = list(model._modules.values())
    if len(modules) > 64:
        raise ValueError("At most 64 operations are supported")
    if not modules or type(modules[0]) is not torch.nn.Linear:
        raise ValueError("The supported Sequential must begin with Linear")
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("Shared modules are not supported")
    dense_modules = [module for module in modules if type(module) is torch.nn.Linear]
    if len(dense_modules) > 16 or any(
        not 1 <= dimension <= 256 for module in dense_modules for dimension in module.weight.shape
    ):
        raise ValueError("At most 16 dense layers with widths 1–256 are supported")
    count = sum(
        module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        for module in dense_modules
    )
    if count > 1_000_000:
        raise ValueError("Model exceeds the 1,000,000 parameter limit")
    activations = {
        torch.nn.ReLU: "relu",
        torch.nn.Tanh: "tanh",
        torch.nn.Sigmoid: "sigmoid",
        torch.nn.Identity: "identity",
        torch.nn.Softmax: "softmax",
    }
    parameters, operations = {}, []
    current = "input"
    for index, module in enumerate(modules):
        name = f"op{index + 1:02}"
        if type(module) is not torch.nn.Linear and type(module) not in activations:
            raise ValueError(f"Unsupported module: {type(module).__name__}")
        if "forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("Forward overrides and hooks are not supported")
        if type(module) is torch.nn.Linear:
            parameters[f"{name}.weight"] = module.weight.detach().cpu().clone()
            bias = f"{name}.bias" if module.bias is not None else None
            if bias:
                parameters[bias] = module.bias.detach().cpu().clone()
            operations.append(
                Dense(
                    id=name, input=current, output=f"{name}.z", weight=f"{name}.weight", bias=bias
                )
            )
        elif type(module) in activations:
            if type(module) is torch.nn.Softmax and module.dim not in (1, -1):
                raise ValueError("Softmax must operate on the feature dimension (1 or -1)")
            operations.append(
                Activation(
                    kind=activations[type(module)], id=name, input=current, output=f"{name}.a"
                )
            )
        else:
            raise ValueError(f"Unsupported module: {type(module).__name__}")
        current = operations[-1].output
    spec = ModelSpec(
        id="sequential",
        name="In-memory Sequential snapshot",
        input_width=modules[0].in_features,
        output_name=current,
        tensors=tuple(
            TensorSpec(id=key, shape=tuple(value.shape)) for key, value in parameters.items()
        ),
        operations=tuple(operations),
    )
    return spec, parameters


def analyze_sequential(
    model: torch.nn.Sequential,
    inputs,
    *,
    baseline_model: torch.nn.Sequential | None = None,
    revision_id: str = "current",
    baseline_revision_id: str | None = None,
    task: Literal["unknown", "regression", "classification"] = "unknown",
) -> dict:
    """Audit an engineer's own supported float32 MLP, without browser model import."""
    spec, parameters = sequential_snapshot(model)
    baseline_spec, baseline = sequential_snapshot(
        baseline_model if baseline_model is not None else model
    )
    if spec != baseline_spec:
        raise ValueError("Comparison requires identical supported architectures")
    spec = ModelSpec.model_validate({**spec.model_dump(), "task": task})
    if isinstance(inputs, torch.Tensor):
        if (
            inputs.ndim != 2
            or not 2 <= inputs.shape[0] <= 256
            or inputs.shape[1] != spec.input_width
        ):
            raise ValueError("Expected 2–256 input rows matching the model's input width")
        inputs = inputs.detach().cpu().tolist()
    return analyze(
        spec,
        parameters,
        baseline,
        inputs,
        revision_id,
        baseline_revision_id or ("baseline" if baseline_model is not None else revision_id),
    )
