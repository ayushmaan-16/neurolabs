import httpx
import pytest
from neurolabs.api import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_draft_preview_and_diagnostics_do_not_commit_and_batch_save_is_atomic(tmp_path):
    app = create_app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        client.headers["x-neurolabs-session"] = (await client.get("/api/v1/session")).json()[
            "token"
        ]
        model = (await client.get("/api/v1/models/curve")).json()
        base = model["head"]
        changes = [
            {"tensor_id": "readout.weight", "index": [0, 0], "value": 1.8},
            {"tensor_id": "readout.bias", "index": [0], "value": 0.2},
        ]
        preview = await client.post(
            "/api/v1/models/curve/draft-preview",
            json={"base_revision": base, "changes": changes, "inputs": [0.5]},
        )
        assert preview.status_code == 200
        assert preview.json()["run"]["outputs"][0][0] == pytest.approx(2.3)
        assert preview.json()["run"]["is_draft"]
        assert preview.json()["run"]["activations"]["hidden.a"][0] == [1.5, 0.5, 0]
        assert len((await client.get("/api/v1/models/curve/revisions")).json()) == 1
        assert (await client.get("/api/v1/models/curve")).json()["head"] == base
        report = (
            await client.post(
                "/api/v1/models/curve/diagnostics",
                json={
                    "revision_id": base,
                    "baseline_revision_id": base,
                    "changes": changes,
                    "inputs": [[0.5], [1], [2]],
                },
            )
        ).json()
        assert report["draft_base_revision_id"] == base
        assert report["revision_id"] == preview.json()["run"]["revision_id"]
        assert len(report["draft_changes"]) == 2
        assert report["summary"]["max_abs_output_change"] == pytest.approx(3.5, abs=1e-6)
        body = {"base_revision": base, "idempotency_key": "grouped", "changes": changes}
        saved = await client.post("/api/v1/models/curve/revisions", json=body)
        assert saved.status_code == 201
        assert len(saved.json()["changes"]) == 2
        assert saved.json()["parent"] == base
        assert (await client.post("/api/v1/models/curve/revisions", json=body)).json()[
            "id"
        ] == saved.json()["id"]
        assert len((await client.get("/api/v1/models/curve/revisions")).json()) == 2
        original = (
            await client.post(
                "/api/v1/models/curve/runs", json={"revision_id": base, "inputs": [[0.5]]}
            )
        ).json()
        assert original["outputs"][0][0] == pytest.approx(0.3)


@pytest.mark.anyio
async def test_invalid_drafts_and_noop_saves_preserve_history(tmp_path):
    app = create_app(tmp_path)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        client.headers["x-neurolabs-session"] = (await client.get("/api/v1/session")).json()[
            "token"
        ]
        base = (await client.get("/api/v1/models/linear")).json()["head"]
        cell = {"tensor_id": "linear.weight", "index": [0, 0], "value": 0.5}
        for changes in ([cell, cell], [{**cell, "index": [0, 2]}], [{**cell, "value": 1e100}]):
            result = await client.post(
                "/api/v1/models/linear/draft-preview",
                json={"base_revision": base, "changes": changes, "inputs": [1]},
            )
            assert result.status_code == 422
        noop = await client.post(
            "/api/v1/models/linear/revisions",
            json={"base_revision": base, "idempotency_key": "noop", "changes": [cell]},
        )
        assert noop.status_code == 422
        assert len((await client.get("/api/v1/models/linear/revisions")).json()) == 1
