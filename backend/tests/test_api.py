import httpx
import pytest
from neurolabs.api import create_app, data_directory


def test_renamed_data_directory_preserves_legacy_data(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROLABS_DATA_DIR", raising=False)
    monkeypatch.delenv("NEUROLENS_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert data_directory() == tmp_path / "neuroLabs"
    legacy = tmp_path / "NeuroLens"
    legacy.mkdir()
    (legacy / "neurolens.sqlite3").touch()
    assert data_directory() == legacy
    monkeypatch.setenv("NEUROLENS_DATA_DIR", str(tmp_path / "legacy-override"))
    assert data_directory() == tmp_path / "legacy-override"
    monkeypatch.setenv("NEUROLABS_DATA_DIR", str(tmp_path / "new-override"))
    assert data_directory() == tmp_path / "new-override"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_api_edit_run_restore_and_origin_checks(tmp_path):
    app = create_app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        token = (await client.get("/api/v1/session")).json()["token"]
        model = (await client.get("/api/v1/models/linear")).json()
        original = model["head"]
        path = "/api/v1/models/linear/runs"
        body = {"revision_id": original, "inputs": [[2]], "captures": ["output"]}
        assert (await client.post(path, json=body)).status_code == 403
        client.headers["x-neurolabs-session"] = token
        result = (await client.post(path, json=body)).json()
        assert result["outputs"] == [[1]]
        assert result["revision_id"] == original
        changed = await client.post(
            "/api/v1/models/linear/revisions",
            json={
                "base_revision": original,
                "idempotency_key": "test",
                "changes": [{"tensor_id": "linear.weight", "index": [0, 0], "value": 3}],
            },
        )
        assert changed.status_code == 201
        revision = changed.json()["id"]
        body["revision_id"] = revision
        assert (await client.post(path, json=body)).json()["outputs"] == [[6]]
        restored = await client.post(
            "/api/v1/models/linear/checkout",
            json={"expected_head": revision, "revision_id": original},
        )
        assert restored.json()["head"] == original
        assert (
            await client.get("/api/v1/models", headers={"Origin": "https://evil.example"})
        ).status_code == 403
        assert (
            await client.get("/api/v1/models", headers={"Host": "evil.example"})
        ).status_code == 400
        assert (await client.get("/api/v1/models/missing")).status_code == 404
