"""Convert the documented sadhaklal/mlp-iris weights without loading downloaded Python.

Download model.safetensors from https://huggingface.co/sadhaklal/mlp-iris first.
Usage: uv run python examples/export_iris_checkpoint.py model.safetensors iris.zip
"""

import argparse

import torch
from neurolabs.packages import export_sequential
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="Downloaded Iris model.safetensors file")
    parser.add_argument("output", help="Destination neuroLabs ZIP package")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = load_file(args.weights, device="cpu")
    expected = {"fc1.weight": (5, 4), "fc1.bias": (5,), "fc2.weight": (3, 5), "fc2.bias": (3,)}
    if set(source) != set(expected) or any(
        tuple(source[key].shape) != shape for key, shape in expected.items()
    ):
        raise ValueError("This converter expects the documented 4 → 5 → 3 Iris checkpoint")
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    model.load_state_dict(
        {
            "0.weight": source["fc1.weight"],
            "0.bias": source["fc1.bias"],
            "2.weight": source["fc2.weight"],
            "2.bias": source["fc2.bias"],
        }
    )
    model.eval()
    path = export_sequential(
        model,
        args.output,
        name="Iris flower classifier",
        task="classification",
        example_inputs=[[0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]],
        feature_names=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        output_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        input_note=(
            "Pretrained sadhaklal/mlp-iris checkpoint. Supply standardized measurements "
            "using the original training mean/std and feature order from its model card. "
            "The included zero/±1 probe inputs are already in standardized feature space. "
            "Outputs are logits. No scaler or accuracy evaluation is included."
        ),
    )
    print(f"Exported {path}: 4 -> 5 -> 3, 43 parameters; source outputs verified on 3 probes.")


if __name__ == "__main__":
    main()
