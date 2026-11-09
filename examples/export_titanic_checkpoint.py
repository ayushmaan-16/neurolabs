"""Package bernardozschaber/titanic-federated-dp-mlp for a manual neuroLabs upload.

See docs/titanic-upload-walkthrough.md. Only JSON, CSV and safetensors are read;
the downloaded repository's Python is not imported or executed.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from neurolabs.packages import export_sequential, inspect_package
from safetensors.torch import load_file

FEATURES = (
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Title",
    "FamilySize",
    "IsAlone",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S",
)


def encode_passengers(path, preprocessing):
    """Match the author's encode(): raw float32, float64 scaling, then float32."""
    if tuple(preprocessing["features"]) != FEATURES:
        raise ValueError("Expected the original Titanic checkpoint's 12-feature order")
    means = np.asarray(preprocessing["scaler"]["mean"], dtype=np.float64)
    stds = np.asarray(preprocessing["scaler"]["std"], dtype=np.float64)
    if (
        means.shape != (12,)
        or stds.shape != (12,)
        or not np.isfinite(means).all()
        or not np.isfinite(stds).all()
        or (stds <= 0).any()
    ):
        raise ValueError("Expected 12 finite scaler means and positive standard deviations")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not 2 <= len(rows) <= 256:
        raise ValueError("Supply 2-256 passenger rows for the diagnostics batch")
    raw = []
    for index, row in enumerate(rows, start=1):
        sex = row["Sex"].strip().lower()
        port = row["Embarked"].strip() or preprocessing["embarked_mode"]
        if sex not in ("male", "female") or port not in "CQS" or len(port) != 1:
            raise ValueError(f"Row {index}: Sex must be male/female and Embarked C/Q/S or blank")
        title = re.search(r" ([A-Za-z]+)\.", row["Name"])
        siblings, parents = float(row["SibSp"]), float(row["Parch"])
        family_size = siblings + parents + 1
        raw.append(
            [
                float(row["Pclass"]),
                float(sex == "female"),
                float(row["Age"] or preprocessing["age_median"]),
                siblings,
                parents,
                float(row["Fare"] or preprocessing["fare_median"]),
                float(preprocessing["titles"].get(title[1] if title else "", 4)),
                family_size,
                float(family_size == 1),
                *[float(port == candidate) for candidate in "CQS"],
            ]
        )
    inputs = ((np.asarray(raw, dtype=np.float32) - means) / stds).astype(np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError("Passenger features must remain finite after preprocessing")
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="Downloaded model.safetensors")
    parser.add_argument("preprocessing", help="Downloaded preprocessing.json")
    parser.add_argument("output", type=Path, help="Destination neuroLabs ZIP")
    parser.add_argument(
        "--passengers",
        type=Path,
        default=Path(__file__).with_name("titanic-passengers.csv"),
        help="Raw passenger CSV; defaults to the included fictional examples",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = load_file(args.weights, device="cpu")
    expected = {
        "net.0.weight": (32, 12),
        "net.0.bias": (32,),
        "net.2.weight": (16, 32),
        "net.2.bias": (16,),
        "net.4.weight": (2, 16),
        "net.4.bias": (2,),
    }
    if set(source) != set(expected) or any(
        tuple(source[key].shape) != shape
        or source[key].dtype != torch.float32
        or not torch.isfinite(source[key]).all()
        for key, shape in expected.items()
    ):
        raise ValueError("Expected the original float32 12 -> 32 -> 16 -> 2 Titanic checkpoint")
    model = torch.nn.Sequential(
        torch.nn.Linear(12, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    ).eval()
    model.load_state_dict({key.removeprefix("net."): value for key, value in source.items()})
    preprocessing = json.loads(Path(args.preprocessing).read_text(encoding="utf-8"))
    inputs = encode_passengers(args.passengers, preprocessing)

    # Independently check the copied layers against the source checkpoint's equations.
    tensor_inputs = torch.from_numpy(inputs)
    with torch.inference_mode():
        reference = tensor_inputs
        for layer in (0, 2, 4):
            reference = reference @ source[f"net.{layer}.weight"].T + source[f"net.{layer}.bias"]
            if layer != 4:
                reference = reference.clamp_min(0)
        outputs = model(tensor_inputs)
    torch.testing.assert_close(outputs, reference, rtol=1e-5, atol=1e-6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = export_sequential(
        model,
        args.output,
        name="Titanic survival MLP",
        task="classification",
        example_inputs=inputs[:16],
        feature_names=FEATURES,
        output_names=["Did not survive (0)", "Survived (1)"],
        input_note=(
            "Source: bernardozschaber/titanic-federated-dp-mlp. "
            "12 -> 32 ReLU -> 16 ReLU -> 2 logits, 978 parameters. "
            "Inputs must use preprocessing.json feature engineering and standardization. "
            "The companion -inputs.csv is already preprocessed; the raw passenger CSV is not. "
            "Default input is the first supplied passenger. Outputs are class scores, not "
            "probabilities; argmax selects 0/1. Included fictional passenger probes do not "
            "measure accuracy or prove a neuron is globally dead."
        ),
    )
    report = inspect_package(path.read_bytes())
    batch_path = path.with_name(f"{path.stem}-inputs.csv")
    np.savetxt(batch_path, inputs, delimiter=",", fmt="%.9g")
    print(f"Exported {path}: 12 -> 32 -> 16 -> 2, {report['parameter_count']} parameters.")
    print(f"Verification: {report['verification']}")
    print(f"Diagnostics CSV (no headers, already standardized): {batch_path}")
    for index, output in enumerate(outputs.tolist(), start=1):
        print(f"Row {index}: logits={output}; predicted class={int(outputs[index - 1].argmax())}")
    print("Default input: " + ", ".join(format(float(value), ".9g") for value in inputs[0]))


if __name__ == "__main__":
    main()
