import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from neurolabs.runtime import execute
from neurolabs.samples import dense_study, samples
from neurolabs.schema import Activation, Dense, ModelSpec, TensorSpec
from onnx import TensorProto, helper, numpy_helper


def test_exact_linear_function():
    spec, values, _ = samples()[0]
    result = execute(spec, values, [[-2], [0], [1], [4]], "baseline", ("output",))
    assert result["outputs"] == [[-1], [0], [0.5], [2]]
    assert result["activations"]["output"] == result["outputs"]
    assert result["revision_id"] == "baseline"
    assert result["input_hash"] != execute(spec, values, [[0]], "baseline")["input_hash"]


def test_xor_truth_table():
    spec, values, _ = samples()[2]
    result = execute(spec, values, [[0, 0], [0, 1], [1, 0], [1, 1]], "baseline")
    assert np.argmax(result["outputs"], axis=1).tolist() == [0, 1, 1, 0]


def test_deep_sample_matches_independent_numpy_composition():
    spec, values, _ = samples()[3]
    inputs = np.array([[-2], [0], [0.5], [2]], dtype=np.float32)
    expected = inputs
    for name in ("stage1", "stage2", "stage3", "readout"):
        expected = expected @ values[f"{name}.weight"].numpy().T + values[f"{name}.bias"].numpy()
        if name != "readout":
            expected = np.tanh(expected)
    result = execute(spec, values, inputs.tolist(), "deep-fixture")
    np.testing.assert_allclose(result["outputs"], expected, atol=1e-6, rtol=1e-5)


def test_dense_sixteen_captures_match_independent_numpy_and_seed_is_stable():
    spec, values, default = dense_study()
    dense = [op for op in spec.operations if isinstance(op, Dense)]
    assert len(dense) == 16
    assert [values[op.weight].shape[0] for op in dense] == [64] * 15 + [2]
    assert sum(value.numel() for key, value in values.items() if key.endswith("weight")) == 57_728
    assert sum(value.numel() for value in values.values()) == 58_690
    for key, value in dense_study()[1].items():
        assert torch.equal(values[key], value)
    inputs = [default, [-1, 0, 0.5, 1], [0, 0, 0, 0]]
    result = execute(
        spec, values, inputs, "dense-study", tuple(op.output for op in spec.operations)
    )
    expected = np.array(inputs, dtype=np.float32)
    for op in spec.operations:
        if isinstance(op, Dense):
            expected = expected @ values[op.weight].numpy().T + values[op.bias].numpy()
        else:
            expected = np.tanh(expected)
        np.testing.assert_allclose(result["activations"][op.output], expected, atol=2e-6, rtol=2e-5)
    assert np.isfinite(result["outputs"]).all()
    assert not np.allclose(result["outputs"][0], result["outputs"][1])
    changed = dict(values)
    changed["layer16.bias"] = values["layer16.bias"] + 0.5
    edited = execute(spec, changed, inputs, "edited")
    np.testing.assert_allclose(np.array(edited["outputs"]) - result["outputs"], 0.5, atol=1e-6)


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "softmax", "identity"])
def test_non_square_execution_matches_onnx(activation):
    spec = ModelSpec(
        id="parity",
        name="Non-square parity",
        input_width=3,
        output_name="output",
        tensors=(TensorSpec(id="w", shape=(2, 3)), TensorSpec(id="b", shape=(2,))),
        operations=(
            Dense(id="dense", input="input", output="z", weight="w", bias="b"),
            Activation(kind=activation, id="activation", input="z", output="output"),
        ),
    )
    weights = np.array([[0.5, -2, 1], [-0.3, 0.7, 2]], dtype=np.float32)
    bias = np.array([0.1, -0.2], dtype=np.float32)
    inputs = np.random.default_rng(42).normal(size=(12, 3)).astype(np.float32)
    operation = {
        "relu": "Relu",
        "tanh": "Tanh",
        "sigmoid": "Sigmoid",
        "softmax": "Softmax",
        "identity": "Identity",
    }[activation]
    attrs = {"axis": -1} if activation == "softmax" else {}
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "w", "b"], ["z"], transB=1),
            helper.make_node(operation, ["z"], ["output"], **attrs),
        ],
        "parity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])],
        [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = session.run(None, {"input": inputs})[0]
    actual = execute(
        spec,
        {"w": torch.from_numpy(weights), "b": torch.from_numpy(bias)},
        inputs.tolist(),
        "baseline",
        ("z", "output"),
    )
    np.testing.assert_allclose(actual["outputs"], expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        actual["activations"]["z"], inputs @ weights.T + bias, atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("inputs", [[], [[1, 2]], [[float("nan")]], [[1e100]], [[1]] * 257])
def test_invalid_inputs(inputs):
    spec, values, _ = samples()[0]
    with pytest.raises(ValueError):
        execute(spec, values, inputs, "baseline")


def test_unknown_capture_and_nonfinite_computation():
    spec, values, _ = samples()[0]
    with pytest.raises(ValueError, match="capture"):
        execute(spec, values, [[1]], "baseline", ("missing",))
    with pytest.raises(ValueError, match="Nonfinite"):
        execute(spec, {"linear.weight": torch.tensor([[3e38]])}, [[3e38]], "baseline")


def test_graph_rejects_disconnected_operations():
    data = samples()[1][0].model_dump()
    data["operations"][1]["input"] = "missing"
    with pytest.raises(ValueError, match="single chain"):
        ModelSpec.model_validate(data)
