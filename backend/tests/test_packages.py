import io
import json
import zipfile

import httpx
import pytest
import torch
from neurolabs.api import create_app
from neurolabs.imports import ImportBusyError, ImportPreviews
from neurolabs.packages import (
    ENTRY_LIMITS,
    Examples,
    Manifest,
    export_sequential,
    inspect_package,
    make_package,
    read_package,
)
from neurolabs.runtime import execute
from neurolabs.samples import samples
from neurolabs.store import Store


@pytest.fixture
def anyio_backend():
    return "asyncio"


def package():
    spec, values, inputs = next(item for item in samples() if item[0].id == "curve")
    manifest = Manifest(spec=spec, default_input=inputs)
    rows = [[-1.0], [0.5], [1.0]]
    examples = Examples(inputs=rows, outputs=execute(spec, values, rows, "test")["outputs"])
    return make_package(manifest, values, examples)


def rewrite(data, filename, value):
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stream, "w") as target:
        for name in source.namelist():
            target.writestr(name, value if name == filename else source.read(name))
    return stream.getvalue()


def test_sequential_export_preserves_values_matches_pytorch_and_does_not_mutate(tmp_path):
    torch.manual_seed(35)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    path = export_sequential(
        model,
        tmp_path / "iris-shaped.zip",
        name="Iris-shaped example",
        example_inputs=[[0.2, -0.7, 1.1, 0.4], [0, 0, 0, 0]],
        feature_names=["a", "b", "c", "d"],
        output_names=["x", "y", "z"],
    )
    report = inspect_package(path.read_bytes())
    assert report["parameter_count"] == 43
    assert report["verification"]["status"] == "passed"
    assert report["verification"]["sample_count"] == 2
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)
    assert model.training
    manifest, parameters, _ = read_package(path.read_bytes())
    assert manifest.metadata.feature_names == ("a", "b", "c", "d")
    assert torch.equal(parameters["op01.weight"], model[0].weight)


def test_incompatible_exports_and_wrong_verification_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported module"):
        export_sequential(
            torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout()), tmp_path / "bad.zip"
        )
    with pytest.raises(ValueError, match="Shape or dtype"):
        export_sequential(torch.nn.Sequential(torch.nn.Linear(2, 1)).double(), tmp_path / "bad.zip")
    broken = rewrite(package(), "examples.json", json.dumps({"inputs": [[0.5]], "outputs": [[99]]}))
    with pytest.raises(ValueError, match="Verification failed"):
        inspect_package(broken)
    wrong_shape = rewrite(
        package(), "examples.json", json.dumps({"inputs": [[0.5]], "outputs": [[0, 0]]})
    )
    with pytest.raises(ValueError, match="shapes"):
        inspect_package(wrong_shape)
    assert not (tmp_path / "bad.zip").exists()


def test_package_limits_path_traversal_schema_weights_and_nonfinite_rejected():
    data = package()
    cases = [b"not a zip"]
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stream, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("../evil.py", "print('never runs')")
    cases.append(stream.getvalue())
    cases.append(rewrite(data, "model.json", b" " * (ENTRY_LIMITS["model.json"] + 1)))
    cases.append(rewrite(data, "weights.safetensors", b"invalid"))
    for item in cases:
        with pytest.raises(ValueError):
            inspect_package(item)
    manifest, values, _ = read_package(data)
    changed = manifest.model_dump()
    changed["spec"]["operations"][1]["kind"] = "conv2d"
    with pytest.raises(ValueError, match="conv2d"):
        inspect_package(rewrite(data, "model.json", json.dumps(changed)))
    values["hidden.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        make_package(manifest, values)


@pytest.mark.anyio
async def test_browser_import_preview_commit_diagnostics_revision_export_and_restart(tmp_path):
    app = create_app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/imports/preview", content=package())
        assert response.status_code == 403
        client.headers["x-neurolabs-session"] = (await client.get("/api/v1/session")).json()[
            "token"
        ]
        before = (await client.get("/api/v1/models")).json()
        preview = await client.post("/api/v1/imports/preview", content=package())
        assert preview.status_code == 200, preview.text
        checked = preview.json()
        assert checked["verification"]["status"] == "passed"
        assert len((await client.get("/api/v1/models")).json()) == len(before)
        endpoint = f"/api/v1/imports/{checked['token']}"
        imported = await client.post(endpoint)
        assert imported.status_code == 201, imported.text
        model = imported.json()
        assert model["spec"]["id"].startswith("import-")
        assert model["spec"]["id"] != "curve"
        assert (await client.post(endpoint)).json()["head"] == model["head"]
        assert len((await client.get("/api/v1/models")).json()) == len(before) + 1
        prefix = f"/api/v1/models/{model['spec']['id']}"
        head = model["head"]
        binary = await client.get(f"{prefix}/revisions/{head}/parameter-data")
        assert binary.status_code == 200
        assert len(binary.content) == 40
        assert binary.headers["x-revision-id"] == head
        sliced = (
            await client.get(f"{prefix}/revisions/{head}/tensors/hidden.weight?row=2&rows=1")
        ).json()
        assert sliced["values"] == [[-1]]
        saved = (
            await client.post(
                f"{prefix}/revisions",
                json={
                    "base_revision": head,
                    "idempotency_key": "import-edit",
                    "name": "Imported experiment",
                    "changes": [{"tensor_id": "hidden.bias", "index": [0], "value": 2}],
                },
            )
        ).json()
        report = (
            await client.post(
                f"{prefix}/diagnostics",
                json={
                    "revision_id": saved["id"],
                    "baseline_revision_id": head,
                    "inputs": [[0.5], [1]],
                },
            )
        ).json()
        assert report["revision_name"] == "Imported experiment"
        assert report["summary"]["max_abs_output_change"] > 0
        exported = await client.get(f"{prefix}/revisions/{saved['id']}/export")
        inspection = inspect_package(exported.content)
        assert inspection["verification"]["status"] == "passed"
        _, values, _ = read_package(exported.content)
        assert values["hidden.bias"][0] == 2
        assert Store(tmp_path).model(model["spec"]["id"])["head"] == saved["id"]
        assert (await client.get("/api/v1/models/curve")).json()["head"] == before[1]["head"]
        invalid = await client.post("/api/v1/imports/preview", content=b"bad zip")
        assert invalid.status_code == 422
        assert (await client.post("/api/v1/imports/not-a-token")).status_code == 404


@pytest.mark.anyio
async def test_wide_export_stays_within_contract_and_slice_bounds(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 256),
        torch.nn.Tanh(),
        torch.nn.Linear(256, 1),
    )
    path = export_sequential(model, tmp_path / "wide.zip", example_inputs=[[0.0] * 256])
    result = inspect_package(path.read_bytes())
    assert result["parameter_count"] == 131841
    assert result["verification"]["status"] == "passed"
    manifest, values, _ = read_package(path.read_bytes())
    app = create_app(tmp_path / "api")
    app.state.store.register(manifest.spec, values, list(manifest.default_input), import_info={})
    model = app.state.store.model(manifest.spec.id)
    prefix = f"/api/v1/models/{manifest.spec.id}/revisions/{model['head']}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        assert (await client.get(f"{prefix}/parameters")).status_code == 422
        binary = await client.get(f"{prefix}/parameter-data")
        assert len(binary.content) == 131841 * 4
        sliced = await client.get(f"{prefix}/tensors/op01.weight?row=192&column=192")
        assert sliced.status_code == 200
        assert len(sliced.json()["values"]) == 64
        assert len(sliced.json()["values"][0]) == 64
        assert (await client.get(f"{prefix}/tensors/op01.weight?rows=65")).status_code == 422
        assert (await client.get(f"{prefix}/tensors/op01.weight?row=256")).status_code == 422


def test_unverified_packages_and_unsupported_visual_topologies(tmp_path):
    manifest, values, _ = read_package(package())
    result = inspect_package(make_package(manifest, values))
    assert result["verification"]["status"] == "not_supplied"
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Tanh())
    with pytest.raises(ValueError, match="at most one activation"):
        export_sequential(model, tmp_path / "bad.zip")
    tied = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    tied[1].weight = tied[0].weight
    with pytest.raises(ValueError, match="Shared parameter"):
        export_sequential(tied, tmp_path / "tied.zip")


def test_worker_timeout_kills_process_releases_slot_and_previews_expire(monkeypatch):
    import subprocess
    import time

    class StuckWorker:
        killed = False

        def communicate(self, data=None, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired("validation", timeout)
            return b"", b""

        def kill(self):
            self.killed = True

    child = StuckWorker()
    monkeypatch.setattr("neurolabs.imports.subprocess.Popen", lambda *args, **kwargs: child)
    previews = ImportPreviews()
    with pytest.raises(ValueError, match="30 second"):
        previews.inspect(package())
    assert child.killed
    assert previews.slots.acquire(blocking=False)
    with pytest.raises(ImportBusyError):
        previews.inspect(package())
    previews.slots.release()
    previews.items["expired"] = (time.monotonic() - 901, b"", {})
    with pytest.raises(KeyError, match="expired"):
        previews.get("expired")


@pytest.mark.anyio
async def test_overflowing_plot_sweep_does_not_hide_a_valid_draft_input(tmp_path):
    app = create_app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        token = (await client.get("/api/v1/session")).json()["token"]
        head = (await client.get("/api/v1/models/linear")).json()["head"]
        response = await client.post(
            "/api/v1/models/linear/draft-preview",
            headers={"x-neurolabs-session": token},
            json={
                "base_revision": head,
                "inputs": [0],
                "changes": [{"tensor_id": "linear.weight", "index": [0, 0], "value": 1.8e38}],
            },
        )
        assert response.status_code == 200
        assert response.json()["run"]["outputs"] == [[0.0]]
        assert response.json()["sweep"] is None
