import hashlib

import httpx
import pytest
from neurolabs.api import create_app
from neurolabs.samples import samples
from neurolabs.schema import CellEdit, EditRequest
from neurolabs.store import Store, canonical


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_names_follow_saved_draft_baseline_and_checkout(tmp_path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(tmp_path)), base_url="http://testserver"
    ) as client:
        client.headers["x-neurolabs-session"] = (await client.get("/api/v1/session")).json()[
            "token"
        ]
        model = (await client.get("/api/v1/models/curve")).json()
        base = model["head"]
        change = {"tensor_id": "hidden.bias", "index": [0], "value": 2}
        body = {"base_revision": base, "idempotency_key": "named", "changes": [change]}
        for invalid in ("x" * 81, "bad\x00name", "two\nlines"):
            assert (
                await client.post("/api/v1/models/curve/revisions", json={**body, "name": invalid})
            ).status_code == 422
        body["name"] = "  Hidden bias experiment  "
        response = await client.post("/api/v1/models/curve/revisions", json=body)
        assert response.status_code == 201
        saved = response.json()
        assert saved["name"] == "Hidden bias experiment"
        assert (await client.post("/api/v1/models/curve/revisions", json=body)).json() == saved
        assert (
            await client.post(
                "/api/v1/models/curve/revisions", json={**body, "name": "Different experiment"}
            )
        ).status_code == 409
        current = (await client.get("/api/v1/models/curve")).json()
        assert current["head_name"] == saved["name"]
        assert current["original_name"] == "Original model"
        history = (await client.get("/api/v1/models/curve/revisions")).json()
        assert history[0]["name"] == saved["name"]
        audit = {"revision_id": saved["id"], "baseline_revision_id": base, "inputs": [[0.5], [1]]}
        report = (await client.post("/api/v1/models/curve/diagnostics", json=audit)).json()
        assert report["revision_name"] == saved["name"]
        assert report["baseline_revision_name"] == "Original model"
        draft = (
            await client.post(
                "/api/v1/models/curve/diagnostics",
                json={
                    **audit,
                    "baseline_revision_id": saved["id"],
                    "changes": [{**change, "value": 3}],
                    "draft_name": "Next experiment",
                },
            )
        ).json()
        assert draft["revision_name"] == "Next experiment"
        assert draft["baseline_revision_name"] == saved["name"]
        assert draft["base_revision_name"] == saved["name"]
        restored = await client.post(
            "/api/v1/models/curve/checkout",
            json={
                "expected_head": saved["id"],
                "revision_id": base,
            },
        )
        assert restored.status_code == 200
        assert (await client.get("/api/v1/models/curve")).json()["head_name"] == "Original model"
        # Blank optional names receive a durable default rather than an empty label.
        unnamed = (
            await client.post(
                "/api/v1/models/curve/revisions",
                json={
                    **body,
                    "idempotency_key": "unnamed",
                    "name": "   ",
                },
            )
        ).json()
        assert unnamed["name"] == f"Revision {unnamed['id'][:8]}"


def test_legacy_database_migration_preserves_artifacts_heads_and_retry(tmp_path):
    store = Store(tmp_path)
    spec, tensors, inputs = next(item for item in samples() if item[0].id == "linear")
    store.register(spec, tensors, inputs)
    base = store.model("linear")["head"]
    request = EditRequest(
        base_revision=base,
        idempotency_key="legacy-retry",
        changes=(CellEdit(tensor_id="linear.weight", index=(0, 0), value=2),),
    )
    saved = store.edit("linear", request)
    artifacts = saved["tensors"]
    # Recreate the actual pre-name six-column revisions table and fingerprint.
    with store.connection() as db:
        db.execute("ALTER TABLE revisions DROP COLUMN name")
        db.execute("PRAGMA user_version = 1")
        fingerprint = hashlib.sha256(canonical(request.model_dump(exclude={"name"})).encode())
        db.execute("UPDATE requests SET fingerprint=?", (fingerprint.hexdigest(),))
    migrated = Store(tmp_path)
    assert migrated.model("linear")["head"] == saved["id"]
    assert migrated.model("linear")["original"] == base
    assert migrated.revision(saved["id"])["tensors"] == artifacts
    assert migrated.edit("linear", request)["id"] == saved["id"]
    assert migrated.revision(base)["name"] == "Original model"
    named = migrated.edit(
        "linear",
        EditRequest(
            base_revision=saved["id"],
            idempotency_key="named",
            name="Persistent name",
            changes=(CellEdit(tensor_id="linear.weight", index=(0, 0), value=3),),
        ),
    )
    reopened = Store(tmp_path)
    assert reopened.model("linear")["head_name"] == "Persistent name"
    assert reopened.revision(named["id"])["name"] == "Persistent name"
    with reopened.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
