"""Translate a bounded, single-chain ONNX MLP into application-owned operations."""

import hashlib
from pathlib import PurePosixPath

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from neurolabs.packages import (
    MAX_PACKAGE_BYTES,
    Examples,
    InputMetadata,
    Manifest,
    inspect_package,
    make_package,
)
from neurolabs.schema import Activation, Dense, ModelSpec, TensorSpec

ACTIVATIONS = {"Relu": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid", "Softmax": "softmax"}
ALLOWED_OPS = {*ACTIVATIONS, "Gemm", "MatMul", "Add", "Identity", "Constant", "Transpose"}
ATTRIBUTE_TYPES = {
    "Gemm": {
        "alpha": onnx.AttributeProto.FLOAT,
        "beta": onnx.AttributeProto.FLOAT,
        "transA": onnx.AttributeProto.INT,
        "transB": onnx.AttributeProto.INT,
    },
    "Softmax": {"axis": onnx.AttributeProto.INT},
    "Constant": {"value": onnx.AttributeProto.TENSOR},
    "Transpose": {"perm": onnx.AttributeProto.INTS},
}


def _attributes(node, allowed):
    names = [item.name for item in node.attribute]
    if len(set(names)) != len(names) or set(names) - set(allowed):
        raise ValueError(f"Unsupported or duplicate attributes on {node.op_type}")
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def _array(tensor):
    # Check before numpy_helper: external tensors must never trigger filesystem reads.
    if tensor.external_data or tensor.data_location == TensorProto.EXTERNAL:
        raise ValueError(
            "External ONNX weights are not supported. Export one file with embedded weights."
        )
    if tensor.data_type != TensorProto.FLOAT:
        raise ValueError("ONNX weights and constants must be float32")
    if not 1 <= len(tensor.dims) <= 2 or any(not 1 <= dim <= 256 for dim in tensor.dims):
        raise ValueError("ONNX weight dimensions must be between 1 and 256 (vectors or matrices)")
    result = numpy_helper.to_array(tensor)
    if not np.isfinite(result).all():
        raise ValueError("ONNX weights must be finite")
    return result.copy()


def _io_shape(value):
    if not value.type.HasField("tensor_type"):
        raise ValueError("ONNX input and output must be float32 tensors")
    tensor = value.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT or len(tensor.shape.dim) != 2:
        raise ValueError("Expected float32 ONNX input/output shaped [batch, features]")
    batch, features = tensor.shape.dim
    if batch.HasField("dim_value") and batch.dim_value != 1:
        raise ValueError("Export ONNX with batch size 1 or a dynamic batch dimension")
    return features.dim_value if features.HasField("dim_value") else None


def convert_onnx(data: bytes, filename="model.onnx"):
    """Return a native ZIP and source provenance; never load custom ops or external files."""
    if not 0 < len(data) <= MAX_PACKAGE_BYTES:
        raise ValueError("ONNX models must be at most 6 MiB")
    try:
        model = onnx.load_model_from_string(data)
    except Exception as exc:
        raise ValueError("Invalid ONNX file. Upload an exported .onnx model.") from exc
    graph = model.graph
    if model.functions or model.training_info or graph.sparse_initializer:
        raise ValueError("ONNX functions, training graphs and sparse parameters are not supported")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("ONNX import requires exactly one input tensor and one output tensor")
    if not 1 <= len(graph.node) <= 128 or len(graph.initializer) > 64:
        raise ValueError("ONNX graph exceeds the supported MLP size (128 nodes / 64 constants)")
    versions = {entry.domain: entry.version for entry in model.opset_import}
    if (
        len(versions) != len(model.opset_import)
        or len(versions) != 1
        or next(iter(versions), None) not in ("", "ai.onnx")
        or not 13 <= next(iter(versions.values()), 0) <= onnx.defs.onnx_opset_version()
    ):
        raise ValueError(
            "Use a standard ONNX opset from 13 through this installation's supported version"
        )
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type not in ALLOWED_OPS:
            qualified = f"{node.domain}:{node.op_type}" if node.domain else node.op_type
            raise ValueError(
                f"Unsupported ONNX operation {qualified}. Import supports sequential dense MLPs; "
                "convolutions, attention and branches are not supported."
            )
        attributes = ATTRIBUTE_TYPES.get(node.op_type, {})
        _attributes(node, attributes)
        for attribute in node.attribute:
            if attribute.type != attributes[attribute.name]:
                raise ValueError(f"Unsupported attribute type on ONNX {node.op_type}")
            if attribute.type == onnx.AttributeProto.TENSOR:
                # The ONNX checker can inspect external Constant files too; reject first.
                _array(attribute.t)
        if len(node.output) != 1 or not node.output[0]:
            raise ValueError("ONNX nodes must have exactly one named output")
    width = _io_shape(graph.input[0])
    declared_output = _io_shape(graph.output[0])
    if width is None or not 1 <= width <= 256:
        raise ValueError("ONNX input feature width must be a known number between 1 and 256")
    constants = {}
    origins = {}
    stored_count = 0

    def add_constant(name, array, origin):
        nonlocal stored_count
        if not name or name in constants:
            raise ValueError("ONNX values must have unique names")
        stored_count += array.size
        if stored_count > 2_000_000:
            raise ValueError("ONNX constants exceed the supported size")
        constants[name], origins[name] = array, origin

    for tensor in graph.initializer:
        add_constant(tensor.name, _array(tensor), tensor.name)
    if graph.input[0].name in constants:
        raise ValueError("ONNX parameters must be embedded initializers, not model inputs")
    # The checker is given a proto, not a path: no external-data resolution is allowed.
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValueError(f"Invalid ONNX graph: {str(exc)[:500]}") from exc

    current = graph.input[0].name
    internal = "input"
    operations, tensor_specs, parameters = [], [], {}
    used_parameters = set()
    used_constants = set()
    dense_count = 0
    pending_matmul = False

    def take_constant(name):
        if name not in constants:
            raise ValueError(
                "Dense weights/biases must be constant; "
                "branches and dynamic weights are unsupported"
            )
        origin = origins[name]
        if origin in used_parameters:
            raise ValueError(
                "Shared ONNX parameters are not supported; export independent dense layers"
            )
        used_parameters.add(origin)
        used_constants.add(origin)
        return constants[name]

    def save_tensor(name, array):
        tensor = torch.from_numpy(np.ascontiguousarray(array).copy())
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Converted ONNX parameters must remain finite float32")
        parameters[name] = tensor
        tensor_specs.append(TensorSpec(id=name, shape=tuple(tensor.shape)))
        if sum(value.numel() for value in parameters.values()) > 1_000_000:
            raise ValueError("Model exceeds the 1,000,000 parameter limit")

    def bias_value(name, size):
        array = take_constant(name)
        if array.shape not in ((size,), (1, size)):
            raise ValueError("ONNX dense bias must have shape [outputs] or [1, outputs]")
        return array.reshape(size)

    for node in graph.node:
        kind, output, inputs = node.op_type, node.output[0], list(node.input)
        if kind == "Constant":
            attributes = _attributes(node, {"value"})
            if (
                inputs
                or set(attributes) != {"value"}
                or not isinstance(attributes["value"], TensorProto)
            ):
                raise ValueError("ONNX Constant must contain a float32 tensor")
            add_constant(output, _array(attributes["value"]), output)
            continue
        if kind in ("Transpose", "Identity") and len(inputs) == 1 and inputs[0] in constants:
            attributes = _attributes(node, {"perm"} if kind == "Transpose" else set())
            array = constants[inputs[0]]
            if kind == "Transpose":
                if array.ndim != 2 or list(attributes.get("perm", [1, 0])) != [1, 0]:
                    raise ValueError(
                        "Only a two-dimensional constant weight transpose is supported"
                    )
                array = array.T
            add_constant(output, array, origins[inputs[0]])
            continue
        if kind == "Identity":
            _attributes(node, set())
            if inputs != [current]:
                raise ValueError("ONNX graph must be a single sequential chain without branches")
            current = output
            continue
        if kind in ("Gemm", "MatMul"):
            if not 2 <= len(inputs) <= (3 if kind == "Gemm" else 2) or inputs[0] != current:
                raise ValueError(
                    "ONNX graph must be a single sequential dense chain without branches"
                )
            attributes = _attributes(
                node, {"alpha", "beta", "transA", "transB"} if kind == "Gemm" else set()
            )
            if (
                attributes.get("alpha", 1.0) != 1.0
                or attributes.get("beta", 1.0) != 1.0
                or attributes.get("transA", 0) != 0
                or attributes.get("transB", 0) not in (0, 1)
            ):
                raise ValueError("Gemm requires alpha=beta=1, transA=0 and transB=0 or 1")
            array = take_constant(inputs[1])
            if array.ndim != 2:
                raise ValueError("Dense ONNX weights must be two-dimensional")
            weight = array if attributes.get("transB", 0) else array.T
            if weight.shape[1] != width:
                raise ValueError("ONNX dense weight dimensions do not match the preceding layer")
            dense_count += 1
            if dense_count > 16:
                raise ValueError("A model must contain between 1 and 16 dense layers")
            op_id = f"op{len(operations) + 1:02}"
            weight_id, bias_id = f"{op_id}.weight", None
            save_tensor(weight_id, weight)
            width = weight.shape[0]
            if len(inputs) == 3 and inputs[2]:
                bias_id = f"{op_id}.bias"
                save_tensor(bias_id, bias_value(inputs[2], width))
            operation = Dense(
                id=op_id, input=internal, output=f"{op_id}.z", weight=weight_id, bias=bias_id
            )
            operations.append(operation)
            internal, current = operation.output, output
            pending_matmul = kind == "MatMul"
        elif kind == "Add":
            _attributes(node, set())
            if len(inputs) != 2 or current not in inputs or not pending_matmul:
                raise ValueError("Add is supported only as the bias immediately after MatMul")
            bias_name = inputs[1] if inputs[0] == current else inputs[0]
            operation = operations[-1]
            bias_id = f"{operation.id}.bias"
            save_tensor(bias_id, bias_value(bias_name, width))
            operations[-1] = operation.model_copy(update={"bias": bias_id})
            current, pending_matmul = output, False
        elif kind in ACTIVATIONS:
            attributes = _attributes(node, {"axis"} if kind == "Softmax" else set())
            if inputs != [current] or not operations or not isinstance(operations[-1], Dense):
                raise ValueError(
                    "Support is limited to one activation after each dense layer, in a single chain"
                )
            if kind == "Softmax" and attributes.get("axis", -1) not in (-1, 1):
                raise ValueError("Softmax must operate on the feature axis (1 or -1)")
            op_id = f"op{len(operations) + 1:02}"
            operation = Activation(
                kind=ACTIVATIONS[kind], id=op_id, input=internal, output=f"{op_id}.a"
            )
            operations.append(operation)
            internal, current, pending_matmul = operation.output, output, False
        else:
            raise ValueError(f"Unsupported placement of ONNX operation {kind}")
    if not operations or current != graph.output[0].name:
        raise ValueError("ONNX output must be the final value in the supported dense chain")
    if declared_output is not None and declared_output != width:
        raise ValueError("ONNX output shape does not match the detected network")
    if set(origins.values()) != used_constants:
        raise ValueError("ONNX has unused parameters or disconnected constant subgraphs")
    filename = PurePosixPath(filename.replace("\\", "/")).name
    name = (
        "".join(c for c in PurePosixPath(filename).stem if c.isprintable()).strip()[:120]
        or "Imported ONNX MLP"
    )
    spec = ModelSpec(
        id="onnx-model",
        name=name,
        input_width=_io_shape(graph.input[0]),
        output_name=internal,
        tensors=tensor_specs,
        operations=operations,
        description=(
            "Dense MLP imported from ONNX. Task and feature meaning require "
            "the model's documentation."
        ),
    )
    # A constant-only graph has already been rejected; execute only the validated operator subset.
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(
            data, sess_options=options, providers=["CPUExecutionProvider"]
        )
        size = spec.input_width
        rng = np.random.default_rng(73)
        probes = np.asarray(
            [
                np.zeros(size),
                np.ones(size),
                -np.ones(size),
                np.linspace(-1, 1, size),
                np.where(np.arange(size) % 2, 0.5, -0.5),
                rng.uniform(-1, 1, size),
            ],
            dtype=np.float32,
        )
        if graph.input[0].type.tensor_type.shape.dim[0].HasField("dim_value"):
            reference = np.concatenate(
                [
                    session.run([graph.output[0].name], {graph.input[0].name: row[None]})[0]
                    for row in probes
                ]
            )
        else:
            reference = session.run([graph.output[0].name], {graph.input[0].name: probes})[0]
    except Exception as exc:
        raise ValueError(f"ONNX Runtime could not verify this model: {str(exc)[:500]}") from exc
    if reference.shape != (6, width) or not np.isfinite(reference).all():
        raise ValueError(
            "ONNX verification must produce finite [batch, outputs] results on generated probes"
        )
    examples = Examples(inputs=probes.tolist(), outputs=reference.tolist())
    manifest = Manifest(
        spec=spec,
        default_input=[0.0] * spec.input_width,
        metadata=InputMetadata(
            input_note=(
                f"Imported from ONNX; input tensor {graph.input[0].name[:100]!r}, "
                f"{spec.input_width} float32 features. "
                "Supply the original preprocessing and feature "
                "order. Feature names, class labels and task are not inferred from the graph. "
                "The default input is zero. The original conversion was checked "
                "against ONNX Runtime "
                "on six generated probes in [-1, 1]; these are not validation data or evidence "
                "of prediction accuracy. Imported parameters are editable layer tensors."
            )
        ),
    )
    package = make_package(manifest, parameters, examples)
    result = inspect_package(package)
    result["source_format"] = "onnx"
    result["source_sha256"] = hashlib.sha256(data).hexdigest()
    result["source_filename"] = filename[:255]
    result["verification"].update(
        method="onnxruntime_probes", reference=f"ONNX Runtime {ort.__version__}"
    )
    return package, result


def make_onnx_example(spec, parameters):
    """Create a self-contained analytic example without using a framework exporter."""
    nodes, initializers = [], []
    width = spec.input_width
    for operation in spec.operations:
        if isinstance(operation, Dense):
            names = [operation.input, operation.weight]
            if operation.bias:
                names.append(operation.bias)
            nodes.append(helper.make_node("Gemm", names, [operation.output], transB=1))
            for name in names[1:]:
                initializers.append(numpy_helper.from_array(parameters[name].numpy(), name=name))
            width = parameters[operation.weight].shape[0]
        else:
            kind = {**{v: k for k, v in ACTIVATIONS.items()}, "identity": "Identity"}[
                operation.kind
            ]
            attributes = {"axis": -1} if kind == "Softmax" else {}
            nodes.append(
                helper.make_node(kind, [operation.input], [operation.output], **attributes)
            )
    graph = helper.make_graph(
        nodes,
        "Analytic rectified curve",
        [
            helper.make_tensor_value_info(
                spec.input_name, TensorProto.FLOAT, ["batch", spec.input_width]
            )
        ],
        [helper.make_tensor_value_info(spec.output_name, TensorProto.FLOAT, ["batch", width])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    return model.SerializeToString()
