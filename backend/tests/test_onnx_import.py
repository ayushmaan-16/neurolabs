import hashlib
import io

import httpx
import numpy as np
import pytest
import torch
from neurolabs.api import create_app
from neurolabs.imports import ImportPreviews
from neurolabs.onnx_import import convert_onnx
from neurolabs.packages import read_package
from neurolabs.runtime import execute
from neurolabs.store import Store
from onnx import TensorProto, helper, numpy_helper


def dense_model(*, matmul=False, activation="Relu", batch="batch", width=3):
    rng = np.random.default_rng(51)
    weights = rng.normal(size=(width, 2)).astype(np.float32)
    bias = rng.normal(size=width).astype(np.float32)
    initializers = [
        numpy_helper.from_array(weights, "weight"),
        numpy_helper.from_array(bias, "bias"),
    ]
    if matmul:
        nodes = [
            helper.make_node("Transpose", ["weight"], ["weight_t"], perm=[1, 0]),
            helper.make_node("MatMul", ["features", "weight_t"], ["product"]),
            helper.make_node("Add", ["bias", "product"], ["linear"]),
        ]
    else:
        nodes = [helper.make_node("Gemm", ["features", "weight", "bias"], ["linear"], transB=1)]
    if activation:
        attributes = {"axis": -1} if activation == "Softmax" else {}
        nodes.append(helper.make_node(activation, ["linear"], ["scores"], **attributes))
    graph = helper.make_graph(
        nodes,
        "test network",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [batch, 2])],
        [
            helper.make_tensor_value_info(
                "scores" if activation else "linear", TensorProto.FLOAT, [batch, width]
            )
        ],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


@pytest.mark.parametrize("matmul", [False, True])
@pytest.mark.parametrize("activation", [None, "Relu", "Tanh", "Sigmoid", "Softmax", "Identity"])
@pytest.mark.parametrize("batch", [1, "batch"])
def test_supported_onnx_graphs_agree_with_runtime(matmul, activation, batch):
    source = dense_model(matmul=matmul, activation=activation, batch=batch)
    raw = source.SerializeToString()
    package, report = convert_onnx(raw, "sample.onnx")
    manifest, parameters, examples = read_package(package)
    assert report["verification"]["status"] == "passed"
    assert report["verification"]["method"] == "onnxruntime_probes"
    assert report["verification"]["sample_count"] == 6
    assert report["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["sha256"] == hashlib.sha256(package).hexdigest()
    assert report["parameter_count"] == 9
    assert manifest.spec.task == "unknown"
    assert manifest.spec.name == "sample"
    np.testing.assert_array_equal(
        parameters["op01.weight"], numpy_helper.to_array(source.graph.initializer[0])
    )
    assert examples is not None


@pytest.mark.parametrize("dynamo", [False, True])
def test_real_pytorch_export_parity_on_unseen_inputs(dynamo, tmp_path):
    torch.manual_seed(37)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 7),
        torch.nn.ReLU(),
        torch.nn.Linear(7, 3),
        torch.nn.Softmax(dim=-1),
    ).eval()
    destination = tmp_path / "pytorch-model.onnx" if dynamo else io.BytesIO()
    torch.onnx.export(
        model,
        (torch.zeros(1, 4),),
        destination,
        dynamo=dynamo,
        external_data=False,
        opset_version=18,
        input_names=["features"],
        output_names=["scores"],
    )
    raw = destination.read_bytes() if dynamo else destination.getvalue()
    package, _ = convert_onnx(raw, "pytorch-model.onnx")
    manifest, values, _ = read_package(package)
    unseen = torch.randn(31, 4)
    actual = execute(manifest.spec, values, unseen.tolist(), "test")["outputs"]
    with torch.inference_mode():
        expected = model(unseen).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_constant_weights_gemm_orientation_and_bias_without_activation():
    model = dense_model(activation=None)
    weights = model.graph.initializer.pop(0)
    weights.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weights).T.copy(), "weight"))
    node = model.graph.node[0]
    del node.attribute[:]
    model.graph.node.insert(0, helper.make_node("Constant", [], ["weight"], value=weights))
    package, report = convert_onnx(model.SerializeToString())
    manifest, _, _ = read_package(package)
    assert report["parameter_count"] == 9
    assert len(manifest.spec.operations) == 1


@pytest.mark.parametrize(
    "case, message",
    [
        ("external", "External ONNX"),
        ("external_constant", "External ONNX"),
        ("unsupported", "Unsupported ONNX operation"),
        ("branch", "single sequential"),
        ("double_activation", "one activation"),
        ("softmax_batch", "feature axis"),
        ("float64", "float32"),
        ("nonfinite", "finite"),
        ("wide", "between 1 and 256"),
        ("batch", "batch size 1"),
        ("dynamic_width", "known number"),
        ("multiple_inputs", "exactly one input"),
        ("functions", "functions"),
        ("scaled_gemm", "alpha=beta=1"),
        ("unused", "unused parameters"),
        ("bias_broadcast", "bias must have shape"),
    ],
)
def test_unsupported_or_unsafe_onnx_is_rejected(case, message):
    model = dense_model()
    graph = model.graph
    if case in ("external", "external_constant"):
        tensor = graph.initializer[0]
        tensor.data_location = TensorProto.EXTERNAL
        tensor.ClearField("raw_data")
        tensor.external_data.add(key="location", value="../../must-never-be-read.safetensors")
        if case == "external_constant":
            node = helper.make_node("Constant", [], ["weight"], value=tensor)
            del graph.initializer[0]
            graph.node.insert(0, node)
    elif case == "unsupported":
        graph.node[1].op_type = "LeakyRelu"
    elif case == "branch":
        graph.node.append(
            helper.make_node("Gemm", ["features", "weight", "bias"], ["branch"], transB=1)
        )
    elif case == "double_activation":
        graph.node.append(helper.make_node("Tanh", ["scores"], ["extra"]))
    elif case == "softmax_batch":
        graph.node[1].op_type = "Softmax"
        graph.node[1].attribute.append(helper.make_attribute("axis", 0))
    elif case == "float64":
        graph.initializer[0].CopyFrom(
            numpy_helper.from_array(np.ones((3, 2), np.float64), "weight")
        )
    elif case == "nonfinite":
        graph.initializer[0].CopyFrom(
            numpy_helper.from_array(np.full((3, 2), np.nan, np.float32), "weight")
        )
    elif case == "wide":
        model = dense_model(width=257)
    elif case == "batch":
        model = dense_model(batch=8)
    elif case == "dynamic_width":
        graph.input[0].type.tensor_type.shape.dim[1].ClearField("dim_value")
        graph.input[0].type.tensor_type.shape.dim[1].dim_param = "features"
    elif case == "multiple_inputs":
        graph.input.append(helper.make_tensor_value_info("another", TensorProto.FLOAT, [1, 2]))
    elif case == "functions":
        model.functions.add(name="CustomFunction", domain="custom")
    elif case == "scaled_gemm":
        graph.node[0].attribute.append(helper.make_attribute("alpha", 2.0))
    elif case == "unused":
        graph.initializer.append(numpy_helper.from_array(np.ones(2, np.float32), "unused"))
    elif case == "bias_broadcast":
        graph.initializer[1].CopyFrom(numpy_helper.from_array(np.ones((3, 1), np.float32), "bias"))
    with pytest.raises(ValueError, match=message):
        convert_onnx(model.SerializeToString())


def test_shared_weights_and_depth_and_parameter_limits():
    for count, width, expected in [
        (2, 2, "Shared ONNX"),
        (17, 2, "16 dense"),
        (16, 256, "1,000,000"),
    ]:
        nodes, initializers = [], []
        previous = "input"
        for index in range(count):
            weight = "shared" if count == 2 else f"w{index}"
            if index == 0 or count != 2:
                initializers.append(
                    numpy_helper.from_array(np.eye(width, dtype=np.float32), weight)
                )
            nodes.append(helper.make_node("MatMul", [previous, weight], [f"y{index}"]))
            previous = f"y{index}"
        graph = helper.make_graph(
            nodes,
            "limits",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, width])],
            initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        with pytest.raises(ValueError, match=expected):
            convert_onnx(model.SerializeToString())


def test_output_disagreement_blocks_conversion(monkeypatch):
    import onnxruntime as ort

    class WrongReference:
        def run(self, names, feed):
            batch = next(iter(feed.values())).shape[0]
            return [np.full((batch, 3), 999.0, dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: WrongReference())
    with pytest.raises(ValueError, match="Verification failed"):
        convert_onnx(dense_model().SerializeToString())


def test_large_onnx_worker_returns_a_bounded_normalized_package():
    rng = np.random.default_rng(72)
    nodes, initializers = [], []
    previous = "features"
    for index in range(16):
        width = 1 if index == 15 else 256
        weights = (rng.normal(size=(width, 256)) / 16).astype(np.float32)
        bias = rng.normal(size=width).astype(np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(weights, f"w{index}"),
                numpy_helper.from_array(bias, f"b{index}"),
            ]
        )
        output = f"linear{index}"
        nodes.append(
            helper.make_node("Gemm", [previous, f"w{index}", f"b{index}"], [output], transB=1)
        )
        if index < 15:
            previous = f"active{index}"
            nodes.append(helper.make_node("Tanh", [output], [previous]))
        else:
            previous = output
    graph = helper.make_graph(
        nodes,
        "maximum width",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 256])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, 1])],
        initializers,
    )
    source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    previews = ImportPreviews()
    report = previews.inspect(source.SerializeToString(), "onnx", "wide.onnx")
    assert report["parameter_count"] == 987_137
    assert report["verification"]["status"] == "passed"
    package, _ = previews.get(report["token"])
    assert 3_000_000 < len(package) < 6 * 1024 * 1024
    manifest, _, _ = read_package(package)
    assert manifest.spec.input_width == 256
    assert len([op for op in manifest.spec.operations if op.kind == "dense"]) == 16


@pytest.mark.anyio
async def test_onnx_preview_commit_edit_diagnostics_export_and_restart(tmp_path):
    app = create_app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://testserver"
    ) as client:
        headers = {"x-neurolabs-session": (await client.get("/api/v1/session")).json()["token"]}
        before = (await client.get("/api/v1/models")).json()
        example = await client.get("/api/v1/import-example.onnx")
        assert example.status_code == 200
        response = await client.post(
            "/api/v1/imports/preview?format=onnx&filename=curve.onnx",
            content=example.content,
            headers=headers,
        )
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["source_format"] == "onnx"
        assert report["verification"]["status"] == "passed"
        assert len((await client.get("/api/v1/models")).json()) == len(before)
        response = await client.post(
            f"/api/v1/imports/{report['token']}",
            headers=headers,
            json={"name": "My ONNX curve", "task": "regression"},
        )
        assert response.status_code == 201, response.text
        imported = response.json()
        assert imported["spec"]["name"] == "My ONNX curve"
        assert imported["spec"]["task"] == "regression"
        assert (
            imported["import_info"]["source_sha256"] == hashlib.sha256(example.content).hexdigest()
        )
        base = f"/api/v1/models/{imported['spec']['id']}"
        result = await client.post(
            base + "/runs",
            headers=headers,
            json={"revision_id": imported["head"], "inputs": [[0.5]]},
        )
        assert result.status_code == 200
        np.testing.assert_allclose(result.json()["outputs"], [[0.3]], atol=1e-6)
        revision = await client.post(
            base + "/revisions",
            headers=headers,
            json={
                "base_revision": imported["head"],
                "idempotency_key": "onnx-edit",
                "name": "Output bias experiment",
                "changes": [{"tensor_id": "op03.bias", "index": [0], "value": 0.5}],
            },
        )
        assert revision.status_code == 201, revision.text
        current = (await client.get(base)).json()
        audit = await client.post(
            base + "/diagnostics",
            headers=headers,
            json={
                "revision_id": current["head"],
                "baseline_revision_id": imported["head"],
                "inputs": [[0], [0.5]],
            },
        )
        assert audit.status_code == 200
        assert audit.json()["summary"]["max_abs_output_change"] > 0
        exported = await client.get(base + f"/revisions/{current['head']}/export")
        manifest, values, _ = read_package(exported.content)
        assert values["op03.bias"].item() == 0.5
        assert manifest.spec.name == "My ONNX curve"
    restarted = Store(tmp_path).model(imported["spec"]["id"])
    assert restarted["import_info"]["verification"]["method"] == "onnxruntime_probes"


@pytest.fixture
def anyio_backend():
    return "asyncio"
