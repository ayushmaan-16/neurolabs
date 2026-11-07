"""Versioned, data-only model contracts. Dense weights are always [output, input]."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,63}$")]
Width = Annotated[int, Field(strict=True, ge=1, le=256)]


class TensorSpec(Contract):
    id: Identifier
    shape: tuple[Width, ...] = Field(min_length=1, max_length=2)
    dtype: Literal["float32"] = "float32"


class Dense(Contract):
    kind: Literal["dense"] = "dense"
    id: Identifier
    input: Identifier
    output: Identifier
    weight: Identifier
    bias: Identifier | None = None


class Activation(Contract):
    kind: Literal["relu", "tanh", "sigmoid", "softmax", "identity"]
    id: Identifier
    input: Identifier
    output: Identifier


Operation = Annotated[Dense | Activation, Field(discriminator="kind")]


class ModelSpec(Contract):
    format_version: Literal[1] = 1
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    input_name: Identifier = "input"
    input_width: Width
    output_name: Identifier
    tensors: tuple[TensorSpec, ...] = Field(min_length=1, max_length=32)
    operations: tuple[Operation, ...] = Field(min_length=1, max_length=64)
    task: Literal["unknown", "regression", "classification"] = "unknown"

    @model_validator(mode="after")
    def validate_graph(self):
        tensors = {t.id: t for t in self.tensors}
        if len(tensors) != len(self.tensors):
            raise ValueError("Tensor identifiers must be unique")
        if len({op.id for op in self.operations}) != len(self.operations):
            raise ValueError("Operation identifiers must be unique")
        width, current = self.input_width, self.input_name
        outputs = {current}
        used: set[str] = set()
        dense_count = 0
        for index, op in enumerate(self.operations):
            if op.input != current or op.output in outputs:
                raise ValueError("Version 1 requires an acyclic single chain with unique outputs")
            if isinstance(op, Dense):
                dense_count += 1
                weight = tensors.get(op.weight)
                if weight is None or len(weight.shape) != 2 or weight.shape[1] != width:
                    raise ValueError(f"Invalid [output, input] weight shape for {op.id}")
                width = weight.shape[0]
                references = [op.weight]
                if op.bias is not None:
                    bias = tensors.get(op.bias)
                    if bias is None or bias.shape != (width,):
                        raise ValueError(f"Invalid bias shape for {op.id}")
                    references.append(op.bias)
                for reference in references:
                    if reference in used:
                        raise ValueError("Shared parameters are not supported in version 1")
                    used.add(reference)
            elif op.kind == "softmax" and index != len(self.operations) - 1:
                raise ValueError("Softmax is only supported as the final operation")
            outputs.add(op.output)
            current = op.output
        if not 1 <= dense_count <= 16:
            raise ValueError("A model must contain between 1 and 16 dense layers")
        if current != self.output_name or used != set(tensors):
            raise ValueError("Output must match the final operation; all tensors must be used")
        count = sum(t.shape[0] * (t.shape[1] if len(t.shape) == 2 else 1) for t in self.tensors)
        if count > 1_000_000:
            raise ValueError("Model exceeds the 1,000,000 parameter limit")
        return self


class CellEdit(Contract):
    tensor_id: Identifier
    index: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(min_length=1, max_length=2)
    value: float


class EditRequest(Contract):
    base_revision: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=100)
    changes: tuple[CellEdit, ...] = Field(min_length=1, max_length=4096)


class RunRequest(Contract):
    revision_id: str = Field(min_length=1, max_length=64)
    inputs: tuple[Annotated[tuple[float, ...], Field(min_length=1, max_length=256)], ...] = Field(
        min_length=1, max_length=256
    )
    captures: tuple[str, ...] = Field(default=(), max_length=64)


class CheckoutRequest(Contract):
    expected_head: str = Field(min_length=1, max_length=64)
    revision_id: str = Field(min_length=1, max_length=64)
