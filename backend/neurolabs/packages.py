"""Versioned native packages: bounded JSON + safetensors, never uploaded Python."""

import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator
from safetensors.torch import load, save

from neurolabs.diagnostics import sequential_snapshot
from neurolabs.runtime import execute, validate_parameters
from neurolabs.schema import Contract, ModelSpec

MAX_PACKAGE_BYTES = 6 * 1024 * 1024
ENTRY_LIMITS = {
    "model.json": 128 * 1024,
    "weights.safetensors": 4_100_000,
    "examples.json": 512 * 1024,
}


class InputMetadata(Contract):
    feature_names: tuple[str, ...] = Field(default=(), max_length=256)
    output_names: tuple[str, ...] = Field(default=(), max_length=256)
    input_note: str = Field(
        default="Supply inputs with the original preprocessing and feature order.", max_length=2000
    )

    @model_validator(mode="after")
    def short_names(self):
        if any(not 1 <= len(name) <= 100 for name in (*self.feature_names, *self.output_names)):
            raise ValueError("Feature/output names must contain 1–100 characters")
        return self


class Manifest(Contract):
    package_version: Literal[1] = 1
    spec: ModelSpec
    default_input: tuple[float, ...] = Field(min_length=1, max_length=256)
    metadata: InputMetadata = Field(default_factory=InputMetadata)

    @model_validator(mode="after")
    def dimensions(self):
        for index, operation in enumerate(self.spec.operations):
            if operation.id == "input":
                raise ValueError("Operation ID 'input' is reserved for graph input neurons")
            if operation.kind != "dense" and (
                index == 0 or self.spec.operations[index - 1].kind != "dense"
            ):
                raise ValueError(
                    "Import supports at most one activation immediately after each dense layer"
                )
        last_dense = next(op for op in reversed(self.spec.operations) if op.kind == "dense")
        width = next(t.shape[0] for t in self.spec.tensors if t.id == last_dense.weight)
        if len(self.default_input) != self.spec.input_width:
            raise ValueError("Default input must match the model input width")
        if (
            self.metadata.feature_names
            and len(self.metadata.feature_names) != self.spec.input_width
        ):
            raise ValueError("Feature names must match the model input width")
        if self.metadata.output_names and len(self.metadata.output_names) != width:
            raise ValueError("Output names must match the model output width")
        if any(abs(value) > torch.finfo(torch.float32).max for value in self.default_input):
            raise ValueError("Default input must remain finite in float32")
        return self


class Examples(Contract):
    inputs: tuple[tuple[float, ...], ...] = Field(min_length=1, max_length=16)
    outputs: tuple[tuple[float, ...], ...] = Field(min_length=1, max_length=16)


def read_package(data: bytes):
    """Read bounded entries in memory. No archive member is extracted to disk."""
    if not 0 < len(data) <= MAX_PACKAGE_BYTES:
        raise ValueError("Model packages must be at most 6 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (
                len(names) != len(set(names))
                or not {"model.json", "weights.safetensors"} <= set(names)
                or not set(names) <= ENTRY_LIMITS.keys()
            ):
                raise ValueError(
                    "ZIP must contain model.json, weights.safetensors "
                    "and optionally examples.json only"
                )
            for entry in entries:
                if (
                    entry.file_size > ENTRY_LIMITS[entry.filename]
                    or entry.flag_bits & 1
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                ):
                    raise ValueError(f"Unsupported or oversized archive entry: {entry.filename}")
            manifest = Manifest.model_validate_json(archive.read("model.json"))
            raw = archive.read("weights.safetensors")
            if len(raw) < 8:
                raise ValueError("Invalid safetensors file")
            header_size = struct.unpack("<Q", raw[:8])[0]
            if not 2 <= header_size <= 64 * 1024 or header_size + 8 > len(raw):
                raise ValueError("Invalid safetensors header size")
            header = json.loads(raw[8 : 8 + header_size])
            expected = {item.id: list(item.shape) for item in manifest.spec.tensors}
            if set(header) - {"__metadata__"} != set(expected):
                raise ValueError("Parameter names do not match the architecture")
            for key, shape in expected.items():
                item = header[key]
                if item.get("shape") != shape or item.get("dtype") != "F32":
                    raise ValueError(f"Expected float32 weights of shape {shape} for {key}")
            parameters = load(raw)
            validate_parameters(manifest.spec, parameters)
            examples = (
                Examples.model_validate_json(archive.read("examples.json"))
                if "examples.json" in names
                else None
            )
    except (
        zipfile.BadZipFile,
        RuntimeError,
        TypeError,
        KeyError,
        UnicodeError,
        struct.error,
    ) as exc:
        raise ValueError(
            "Invalid model package; export architecture and float32 weights together"
        ) from exc
    return manifest, parameters, examples


def inspect_package(data: bytes) -> dict:
    manifest, parameters, examples = read_package(data)
    spec = manifest.spec
    # Verify a default forward pass even when no source-model examples were provided.
    execute(spec, parameters, [manifest.default_input], "validation")
    verification = {"status": "not_supplied", "sample_count": 0, "max_abs_error": None}
    if examples:
        actual = execute(spec, parameters, examples.inputs, "verification")["outputs"]
        if len(actual) != len(examples.outputs) or any(
            len(row) != len(reference)
            for row, reference in zip(actual, examples.outputs, strict=True)
        ):
            raise ValueError("Verification output shapes do not match the model")
        pairs = [
            (a, b)
            for row, reference in zip(actual, examples.outputs, strict=True)
            for a, b in zip(row, reference, strict=True)
        ]
        if not all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5) for a, b in pairs):
            raise ValueError(
                "Verification failed: neuroLabs outputs differ from the supplied outputs"
            )
        verification = {
            "status": "passed",
            "sample_count": len(actual),
            "max_abs_error": max(abs(a - b) for a, b in pairs),
        }
    return {
        "manifest": manifest.model_dump(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "parameter_count": sum(value.numel() for value in parameters.values()),
        "verification": verification,
    }


def make_package(manifest: Manifest, parameters: dict, examples: Examples | None = None) -> bytes:
    validate_parameters(manifest.spec, parameters)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.json", manifest.model_dump_json(indent=2))
        archive.writestr("weights.safetensors", save(parameters))
        if examples:
            archive.writestr("examples.json", examples.model_dump_json(indent=2))
    data = stream.getvalue()
    if len(data) > MAX_PACKAGE_BYTES:
        raise ValueError("Export exceeds the 6 MiB package limit")
    return data


def export_sequential(
    model: torch.nn.Sequential,
    path: str | Path,
    *,
    name: str = "My MLP",
    task: Literal["unknown", "regression", "classification"] = "unknown",
    example_inputs=None,
    default_input=None,
    feature_names=(),
    output_names=(),
    input_note: str = "Supply inputs with the original preprocessing and feature order.",
) -> Path:
    """Export a caller-owned exact Sequential; no model download or Python deserialization."""
    spec, parameters = sequential_snapshot(model)
    source_parameters = [value for _, value in model.named_parameters(remove_duplicate=False)]
    storage_ids = [
        (value.device, value.untyped_storage().data_ptr()) for value in source_parameters
    ]
    if len(set(storage_ids)) != len(storage_ids):
        raise ValueError("Shared parameter storage is not supported by the exporter")
    spec = ModelSpec.model_validate({**spec.model_dump(), "name": name, "task": task})
    validate_parameters(spec, parameters)
    examples = None
    if example_inputs is not None:
        x = torch.as_tensor(example_inputs, dtype=torch.float32, device="cpu")
        if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] != spec.input_width:
            raise ValueError("Provide 1–16 example rows with the model's input width")
        # Execute the caller's supported modules independently of the neuroLabs runtime.
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise ValueError("Move the model to CPU before exporting verification examples")
        with torch.inference_mode():
            original_output = model(x).tolist()
        examples = Examples(inputs=x.tolist(), outputs=original_output)
    manifest = Manifest(
        spec=spec,
        default_input=(
            default_input
            if default_input is not None
            else examples.inputs[0]
            if examples
            else [0.0] * spec.input_width
        ),
        metadata=InputMetadata(
            feature_names=feature_names, output_names=output_names, input_note=input_note
        ),
    )
    data = make_package(manifest, parameters, examples)
    inspect_package(data)
    destination = Path(path)
    destination.write_bytes(data)
    return destination
