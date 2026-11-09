"""Deterministic, untrained upload fixtures used by browser tests."""

from pathlib import Path

import torch
from neurolabs.packages import export_sequential

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
