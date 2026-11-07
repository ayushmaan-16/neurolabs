"""Analytic fixtures and seeded visualization studies; none are trained models."""

import torch

from neurolabs.schema import Activation, Dense, ModelSpec, TensorSpec


def dense_study() -> tuple[ModelSpec, dict[str, torch.Tensor], list[float]]:
    """Use the 16-dense-layer contract limit without changing global RNG state."""
    generator = torch.Generator().manual_seed(16064)
    parameters = {}
    operations = []
    current, width = "input", 4
    for index in range(16):
        name = f"layer{index + 1:02}"
        output_width = 64 if index < 15 else 2
        parameters[f"{name}.weight"] = torch.randn(output_width, width, generator=generator) * (
            1.25 / width**0.5
        )
        parameters[f"{name}.bias"] = torch.randn(output_width, generator=generator) * 0.05
        operations.append(
            Dense(
                id=name,
                input=current,
                output=f"{name}.z",
                weight=f"{name}.weight",
                bias=f"{name}.bias",
            )
        )
        current = f"{name}.z"
        if index < 15:
            operations.append(
                Activation(kind="tanh", id=f"{name}.tanh", input=current, output=f"{name}.a")
            )
            current = f"{name}.a"
        width = output_width
    return (
        ModelSpec(
            id="dense16",
            name="Dense sixteen",
            description=(
                "16 dense layers · 64 neurons in each of 15 hidden layers. "
                "Seeded Tanh study, not a trained predictor."
            ),
            input_width=4,
            output_name=current,
            tensors=tuple(
                TensorSpec(id=key, shape=tuple(value.shape)) for key, value in parameters.items()
            ),
            operations=tuple(operations),
        ),
        parameters,
        [0.5, -0.25, 0.75, 0.1],
    )


def samples() -> list[tuple[ModelSpec, dict[str, torch.Tensor], list[float]]]:
    linear = ModelSpec(
        id="linear",
        name="The linear map",
        description="An exact starting point: f(x) = 0.5x.",
        input_width=1,
        output_name="output",
        task="regression",
        tensors=(TensorSpec(id="linear.weight", shape=(1, 1)),),
        operations=(Dense(id="linear", input="input", output="output", weight="linear.weight"),),
    )
    curve = ModelSpec(
        id="curve",
        name="A rectified curve",
        description="Three ReLU features form a piecewise-linear curve. Inspect each contribution.",
        input_width=1,
        output_name="output",
        task="regression",
        tensors=(
            TensorSpec(id="hidden.weight", shape=(3, 1)),
            TensorSpec(id="hidden.bias", shape=(3,)),
            TensorSpec(id="readout.weight", shape=(1, 3)),
            TensorSpec(id="readout.bias", shape=(1,)),
        ),
        operations=(
            Dense(
                id="hidden",
                input="input",
                output="hidden.z",
                weight="hidden.weight",
                bias="hidden.bias",
            ),
            Activation(kind="relu", id="relu", input="hidden.z", output="hidden.a"),
            Dense(
                id="readout",
                input="hidden.a",
                output="output",
                weight="readout.weight",
                bias="readout.bias",
            ),
        ),
    )
    xor = ModelSpec(
        id="xor",
        name="The XOR classifier",
        description="A hand-built two-feature classifier. Outputs are logits, not probabilities.",
        input_width=2,
        output_name="output",
        task="classification",
        tensors=(
            TensorSpec(id="hidden.weight", shape=(2, 2)),
            TensorSpec(id="hidden.bias", shape=(2,)),
            TensorSpec(id="readout.weight", shape=(2, 2)),
            TensorSpec(id="readout.bias", shape=(2,)),
        ),
        operations=(
            Dense(
                id="hidden",
                input="input",
                output="hidden.z",
                weight="hidden.weight",
                bias="hidden.bias",
            ),
            Activation(kind="relu", id="relu", input="hidden.z", output="hidden.a"),
            Dense(
                id="readout",
                input="hidden.a",
                output="output",
                weight="readout.weight",
                bias="readout.bias",
            ),
        ),
    )
    deep_parameters = {
        "stage1.weight": torch.tensor([[1.0], [-0.8], [0.5]]),
        "stage1.bias": torch.tensor([0.2, 0.3, -0.1]),
        "stage2.weight": torch.tensor([[0.6, -0.3, 0.2], [-0.4, 0.8, 0.5], [0.3, 0.2, -0.7]]),
        "stage2.bias": torch.tensor([0.1, 0.1, 0.1]),
        "stage3.weight": torch.tensor([[0.7, -0.2, 0.4], [-0.5, 0.6, 0.3]]),
        "stage3.bias": torch.tensor([0.1, -0.1]),
        "readout.weight": torch.tensor([[0.8, -0.6]]),
        "readout.bias": torch.tensor([0.05]),
    }
    deep_ops = []
    current = "input"
    for name in ("stage1", "stage2", "stage3", "readout"):
        deep_ops.append(
            Dense(
                id=name,
                input=current,
                output=f"{name}.z",
                weight=f"{name}.weight",
                bias=f"{name}.bias",
            )
        )
        current = f"{name}.z"
        if name != "readout":
            deep_ops.append(
                Activation(kind="tanh", id=f"{name}.tanh", input=current, output=f"{name}.a")
            )
            current = f"{name}.a"
    deep = ModelSpec(
        id="deep",
        name="Deep signal study",
        input_width=1,
        output_name=current,
        task="regression",
        description="Five neuron layers, twenty connections, and Tanh activations.",
        tensors=tuple(
            TensorSpec(id=key, shape=tuple(value.shape)) for key, value in deep_parameters.items()
        ),
        operations=tuple(deep_ops),
    )
    return [
        (linear, {"linear.weight": torch.tensor([[0.5]])}, [1.0]),
        (
            curve,
            {
                "hidden.weight": torch.tensor([[1.0], [1.0], [-1.0]]),
                "hidden.bias": torch.tensor([1.0, 0.0, 0.0]),
                "readout.weight": torch.tensor([[0.8, -1.2, 0.4]]),
                "readout.bias": torch.tensor([-0.3]),
            },
            [0.5],
        ),
        (
            xor,
            {
                "hidden.weight": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
                "hidden.bias": torch.tensor([0.0, -1.0]),
                "readout.weight": torch.tensor([[-2.0, 4.0], [2.0, -4.0]]),
                "readout.bias": torch.tensor([1.0, -1.0]),
            },
            [0.0, 1.0],
        ),
        (deep, deep_parameters, [0.5]),
        dense_study(),
    ]
