import copy
import json

import httpx
import numpy as np
import pytest
import torch
from neurolabs.api import create_app
from neurolabs.diagnostics import analyze, analyze_sequential
from neurolabs.samples import samples


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_linear_comparison_matches_closed_form_and_fingerprints():
    spec, baseline, _ = samples()[0]
    current = {"linear.weight": torch.tensor([[1.5]])}
    result = analyze(spec, current, baseline, [[-2], [0], [2]], "edited", "original")
    assert result["summary"]["mean_abs_output_change"] == pytest.approx(4 / 3)
    assert result["summary"]["max_abs_output_change"] == 2
    assert result["summary"]["argmax_flips"] is None
    layer = result["layers"][0]
    assert layer["delta_rms"] == pytest.approx(np.sqrt(8 / 3))
    assert layer["baseline_rms"] == pytest.approx(np.sqrt(2 / 3))
    assert layer["neurons"][0]["zero_fraction"] == pytest.approx(1 / 3)
    assert result["parameter_hash"] != result["baseline_parameter_hash"]
    identical = analyze(spec, baseline, baseline, [[-2], [0], [2]], "original", "original")
    assert identical["input_hash"] == result["input_hash"]
    assert identical["parameter_hash"] == identical["baseline_parameter_hash"]
    assert identical["summary"]["max_abs_output_change"] == 0


def test_inactivity_is_scoped_to_supplied_batch():
    spec, values, _ = samples()[1]
    positive = analyze(spec, values, values, [[0.5], [1], [2]], "a", "a")
    assert positive["summary"]["inactive_relu_neurons"] == 1
    neuron = positive["layers"][0]["neurons"][2]
    assert neuron["inactive_on_batch"]
    assert neuron["zero_fraction"] == 1
    negative = analyze(spec, values, values, [[-2], [-1]], "a", "a")
    assert not negative["layers"][0]["neurons"][2]["inactive_on_batch"]
    assert "batch" in positive["scope"]


@pytest.mark.parametrize("activation", [torch.nn.Tanh, torch.nn.Sigmoid])
def test_saturation_is_measured_from_activations(activation):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), activation())
    with torch.no_grad():
        model[0].weight.fill_(1)
    result = analyze_sequential(model, [[-10], [10], [0]])
    assert result["layers"][0]["neurons"][0]["saturation_fraction"] == pytest.approx(2 / 3)
    assert result["summary"]["inactive_relu_neurons"] == 0


def test_engineers_sequential_matches_pytorch_and_preserves_owned_models():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2))
    reference = copy.deepcopy(model)
    with torch.no_grad():
        model[2].bias[0] += 0.75
    inputs = torch.tensor([[-1.0, 0.5], [0.3, 1.0], [0.0, 0.0]])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = analyze_sequential(model, inputs, baseline_model=reference)
    np.testing.assert_allclose(
        [case["current_output"] for case in result["cases"]],
        model(inputs).detach().numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [case["baseline_output"] for case in result["cases"]],
        reference(inputs).detach().numpy(),
        atol=1e-6,
    )
    assert result["summary"]["max_abs_output_change"] == pytest.approx(0.75, abs=1e-6)
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert torch.equal(rng, torch.get_rng_state())


def test_argmax_changes_are_not_accuracy():
    model = torch.nn.Sequential(torch.nn.Linear(1, 2))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.copy_(torch.tensor([1.0, -1.0]))
    reference = copy.deepcopy(model)
    with torch.no_grad():
        model[0].bias.copy_(torch.tensor([-1.0, 1.0]))
    result = analyze_sequential(model, [[0], [1]], baseline_model=reference, task="classification")
    assert result["summary"]["argmax_flips"] == 2
    assert "accuracy" not in result["summary"]


@pytest.mark.parametrize(
    "model",
    [
        torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Dropout()),
        torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softmax(dim=0)),
        torch.nn.Sequential(torch.nn.Linear(257, 2)),
        torch.nn.Sequential(torch.nn.Linear(1, 2).double()),
    ],
)
def test_unsupported_modules_shapes_and_dtypes_are_rejected(model):
    with pytest.raises(ValueError):
        analyze_sequential(model, [[0], [1]])


def test_mismatched_architecture_and_shared_modules_are_rejected():
    with pytest.raises(ValueError, match="identical"):
        analyze_sequential(
            torch.nn.Sequential(torch.nn.Linear(1, 1)),
            [[0], [1]],
            baseline_model=torch.nn.Sequential(torch.nn.Linear(1, 2)),
        )
    linear = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Shared"):
        analyze_sequential(torch.nn.Sequential(linear, torch.nn.ReLU(), linear), [[0, 0], [1, 1]])


def test_large_finite_values_produce_finite_float64_statistics():
    spec, _, _ = samples()[0]
    values = {"linear.weight": torch.tensor([[1e30]])}
    result = analyze(spec, values, values, [[1], [2]], "a", "a")
    assert result["layers"][0]["rms"] > 1e30
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="2–256"):
        analyze(spec, values, values, [[1]], "a", "a")


def test_python_bridge_rejects_behavior_overrides_and_input_shapes():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    handle = model[0].register_forward_hook(lambda module, inputs, output: output + 1)
    with pytest.raises(ValueError, match="hooks"):
        analyze_sequential(model, [[0], [1]])
    handle.remove()
    model.forward = lambda inputs: inputs
    with pytest.raises(ValueError, match="overrides"):
        analyze_sequential(model, [[0], [1]])
    plain = torch.nn.Sequential(torch.nn.Linear(1, 1))
    for inputs in (torch.zeros(257, 1), torch.zeros(2, 2), torch.zeros(2)):
        with pytest.raises(ValueError, match="input rows"):
            analyze_sequential(plain, inputs)


def test_consecutive_activations_are_audited_at_each_visual_stage():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1, bias=False), torch.nn.Tanh(), torch.nn.Sigmoid()
    )
    with torch.no_grad():
        model[0].weight.fill_(1)
    inputs = torch.tensor([[-10.0], [10.0]])
    result = analyze_sequential(model, inputs)
    assert [layer["activation"] for layer in result["layers"]] == ["tanh", "sigmoid"]
    assert result["layers"][0]["neurons"][0]["saturation_fraction"] == 1
    assert result["layers"][1]["neurons"][0]["saturation_fraction"] == 0
    np.testing.assert_allclose(
        [case["current_output"] for case in result["cases"]],
        model(inputs).detach().numpy(),
        atol=1e-6,
    )


@pytest.mark.anyio
async def test_diagnostic_api_is_read_only_bounded_and_model_scoped(tmp_path):
    app = create_app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        model = (await client.get("/api/v1/models/curve")).json()
        body = {
            "revision_id": model["head"],
            "baseline_revision_id": model["original"],
            "inputs": [[0.5], [1]],
        }
        route = "/api/v1/models/curve/diagnostics"
        assert (await client.post(route, json=body)).status_code == 403
        client.headers["x-neurolabs-session"] = (await client.get("/api/v1/session")).json()[
            "token"
        ]
        response = await client.post(route, json=body)
        assert response.status_code == 200
        assert response.json()["summary"]["inactive_relu_neurons"] == 1
        assert (await client.get("/api/v1/models/curve")).json()["head"] == model["head"]
        assert len((await client.get("/api/v1/models/curve/revisions")).json()) == 1
        other = (await client.get("/api/v1/models/linear")).json()
        assert (
            await client.post(route, json={**body, "baseline_revision_id": other["head"]})
        ).status_code == 422
        assert (await client.post(route, json={**body, "inputs": [[1]] * 257})).status_code == 422
        assert (
            await client.post(route, json={**body, "inputs": [[1, 2], [3, 4]]})
        ).status_code == 422
