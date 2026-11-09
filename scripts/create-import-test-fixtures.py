"""Deterministic, untrained upload fixtures used by browser tests."""

from pathlib import Path

import torch
from neurolabs.onnx_import import make_onnx_example
from neurolabs.packages import export_sequential
from neurolabs.samples import samples
from onnx import TensorProto, helper

torch.set_num_threads(1)
torch.manual_seed(73)
destination = Path(".local")
destination.mkdir(exist_ok=True)
small = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
export_sequential(
    small,
    destination / "import-small.zip",
    name="Upload test MLP",
    task="classification",
    example_inputs=[[0.2, -0.7, 1.1, 0.4], [0, 0, 0, 0]],
    feature_names=["a", "b", "c", "d"],
    output_names=["red", "green", "blue"],
    input_note="Untrained test fixture. Supply four numeric features in order a, b, c, d.",
)
modules = []
for _ in range(15):
    modules.extend([torch.nn.Linear(256, 256), torch.nn.Tanh()])
modules.append(torch.nn.Linear(256, 1))
large = torch.nn.Sequential(*modules)
export_sequential(
    large,
    destination / "import-large.zip",
    name="Maximum import fixture",
    example_inputs=[[0.0] * 256],
)
print("Created small (43 parameters) and large (987,137 parameters) test packages.")
curve_spec, curve_values, _ = next(item for item in samples() if item[0].id == "curve")
(destination / "import-curve.onnx").write_bytes(make_onnx_example(curve_spec, curve_values))
unsupported = helper.make_model(
    helper.make_graph(
        [helper.make_node("LeakyRelu", ["x"], ["y"])],
        "unsupported",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])],
    ),
    opset_imports=[helper.make_opsetid("", 18)],
    ir_version=10,
)
(destination / "import-unsupported.onnx").write_bytes(unsupported.SerializeToString())
