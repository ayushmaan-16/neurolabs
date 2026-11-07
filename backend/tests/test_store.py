from concurrent.futures import ThreadPoolExecutor

import pytest
from neurolabs.runtime import execute
from neurolabs.samples import samples
from neurolabs.schema import CellEdit, EditRequest
from neurolabs.store import ConflictError, Store


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path)
    for spec, values, default_input in samples():
        result.register(spec, values, default_input)
    return result


def request(base, key="first", value=2.0):
    return EditRequest(
        base_revision=base,
        idempotency_key=key,
        changes=(CellEdit(tensor_id="linear.weight", index=(0, 0), value=value),),
    )


def output(store, revision):
    spec, parameters = store.snapshot("linear", revision)
    return execute(spec, parameters, [[2]], revision)["outputs"]


def test_edit_restore_branch_and_restart_preserve_original(store):
    original = store.model("linear")["head"]
    changed = store.edit("linear", request(original))["id"]
    assert output(store, original) == [[1]]
    assert output(store, changed) == [[4]]
    store.checkout("linear", changed, original)
    branch = store.edit("linear", request(original, "branch", -1))["id"]
    restarted = Store(store.root)
    assert restarted.model("linear")["head"] == branch
    assert output(restarted, changed) == [[4]]
    assert output(restarted, branch) == [[-2]]
    assert len(restarted.history("linear")) == 3


def test_idempotency_and_conflicts(store):
    original = store.model("linear")["head"]
    change = request(original)
    first = store.edit("linear", change)
    assert store.edit("linear", change) == first
    with pytest.raises(ConflictError, match="Idempotency"):
        store.edit("linear", request(original, value=3))
    with pytest.raises(ConflictError, match="Active revision"):
        store.edit("linear", request(original, "stale"))


def test_two_simultaneous_edits_only_one_wins(store):
    original = store.model("linear")["head"]

    def attempt(key):
        try:
            return store.edit("linear", request(original, key))["id"]
        except ConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["a", "b"]))
    assert sum(result is not None for result in results) == 1
    assert len(store.history("linear")) == 2


def test_invalid_second_cell_rolls_back_entire_edit(store):
    original = store.model("linear")["head"]
    changes = (
        CellEdit(tensor_id="linear.weight", index=(0, 0), value=7),
        CellEdit(tensor_id="linear.weight", index=(0, 1), value=8),
    )
    with pytest.raises(ValueError, match="bounds"):
        store.edit(
            "linear", EditRequest(base_revision=original, idempotency_key="bad", changes=changes)
        )
    assert store.model("linear")["head"] == original
    assert output(store, original) == [[1]]
    assert len(store.history("linear")) == 1


def test_float32_overflow_and_cross_model_restore_rejected(store):
    original = store.model("linear")["head"]
    with pytest.raises(ValueError, match="finite"):
        store.edit("linear", request(original, value=1e100))
    with pytest.raises(ValueError, match="another model"):
        store.checkout("linear", original, store.model("xor")["head"])


def test_artifact_integrity_is_checked(store):
    original = store.model("linear")["head"]
    digest = store.revision(original)["tensors"]["linear.weight"]
    (store.blobs / f"{digest}.safetensors").write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        store.snapshot("linear", original)


def test_legacy_database_keeps_saved_revisions(store):
    original = store.model("linear")["head"]
    changed = store.edit("linear", request(original))["id"]
    store.database_path.rename(store.root / "neurolens.sqlite3")
    renamed = Store(store.root)
    assert renamed.model("linear")["head"] == changed
    assert output(renamed, original) == [[1]]
    assert output(renamed, changed) == [[4]]
