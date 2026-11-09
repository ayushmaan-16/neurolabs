"""Shared atomic validation for saved revisions and read-only draft previews."""

import hashlib
import json

import torch

from neurolabs.runtime import validate_parameters


def apply_changes(spec, parameters, edits):
    result = dict(parameters)
    seen, copied, changes = set(), set(), []
    for edit in edits:
        coordinate = (edit.tensor_id, edit.index)
        if coordinate in seen:
            raise ValueError("An edit may only address each coordinate once")
        seen.add(coordinate)
        tensor = result.get(edit.tensor_id)
        if (
            tensor is None
            or len(edit.index) != tensor.ndim
            or any(i >= size for i, size in zip(edit.index, tensor.shape, strict=True))
        ):
            raise ValueError("Tensor coordinate is out of bounds")
        effective = torch.tensor(edit.value, dtype=torch.float32)
        if not torch.isfinite(effective).item():
            raise ValueError("Edit must remain finite in float32")
        old = tensor[edit.index].item()
        if old == effective.item():
            continue
        if edit.tensor_id not in copied:
            tensor = tensor.clone()
            result[edit.tensor_id] = tensor
            copied.add(edit.tensor_id)
        tensor[edit.index] = effective
        changes.append(
            {
                "tensor_id": edit.tensor_id,
                "index": list(edit.index),
                "old": old,
                "requested": edit.value,
                "value": effective.item(),
            }
        )
    validate_parameters(spec, result)
    return result, changes


def draft_id(base_revision, changes):
    effective = sorted(
        (
            {"tensor_id": item["tensor_id"], "index": item["index"], "value": item["value"]}
            for item in changes
        ),
        key=lambda item: (item["tensor_id"], item["index"]),
    )
    content = json.dumps([base_revision, effective], sort_keys=True, separators=(",", ":"))
    return "draft-" + hashlib.sha256(content.encode()).hexdigest()[:32]
