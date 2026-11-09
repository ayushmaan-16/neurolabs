"""Immutable tensor artifacts and transactional revision metadata."""

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load, save

from neurolabs.editing import apply_changes
from neurolabs.runtime import validate_parameters
from neurolabs.schema import EditRequest, ModelSpec


class ConflictError(ValueError):
    """The expected workspace state no longer matches."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.database_path = root / "neurolabs.sqlite3"
        legacy_database = root / "neurolens.sqlite3"
        if not self.database_path.exists() and legacy_database.is_file():
            self.database_path = legacy_database
        self.blobs = root / "tensors"
        self.blobs.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY, spec TEXT NOT NULL,
                    original TEXT NOT NULL, head TEXT NOT NULL, default_input TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id),
                    parent TEXT, tensors TEXT NOT NULL, changes TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    model_id TEXT NOT NULL, key TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    revision_id TEXT NOT NULL REFERENCES revisions(id), PRIMARY KEY(model_id, key)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, revision_id TEXT NOT NULL REFERENCES revisions(id),
                    result TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(revisions)")}
            if "name" not in columns:
                db.execute("ALTER TABLE revisions ADD COLUMN name TEXT NOT NULL DEFAULT ''")
            db.execute("""UPDATE revisions SET name = CASE WHEN parent IS NULL THEN 'Original model'
                ELSE 'Revision ' || substr(id, 1, 8) END WHERE name = ''""")
            db.execute("PRAGMA user_version = 2")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def write_tensor(self, tensor: torch.Tensor) -> str:
        data = save({"value": tensor.detach().cpu().contiguous()})
        digest = hashlib.sha256(data).hexdigest()
        path = self.blobs / f"{digest}.safetensors"
        if not path.exists():
            temporary = self.blobs / f"{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return digest

    def read_tensor(self, digest: str) -> torch.Tensor:
        data = (self.blobs / f"{digest}.safetensors").read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Tensor artifact failed integrity verification")
        return load(data)["value"]

    def register(self, spec: ModelSpec, parameters: dict, default_input: list[float]) -> None:
        validate_parameters(spec, parameters)
        blobs = {key: self.write_tensor(value) for key, value in parameters.items()}
        revision = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM models WHERE id=?", (spec.id,)).fetchone():
                return
            db.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?)",
                (spec.id, spec.model_dump_json(), revision, revision, canonical(default_input)),
            )
            db.execute(
                "INSERT INTO revisions (id, model_id, parent, tensors, changes, created_at, name) "
                "VALUES (?, ?, NULL, ?, '[]', ?, ?)",
                (
                    revision,
                    spec.id,
                    canonical(blobs),
                    datetime.now(UTC).isoformat(),
                    "Original model",
                ),
            )

    def models(self) -> list[dict]:
        with self.connection() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM models ORDER BY rowid")]
        return [self.model(model_id) for model_id in ids]

    def model(self, model_id: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                """SELECT models.*, head_revision.name AS head_name,
                original_revision.name AS original_name FROM models
                JOIN revisions AS head_revision ON head_revision.id = models.head
                JOIN revisions AS original_revision ON original_revision.id = models.original
                WHERE models.id=?""",
                (model_id,),
            ).fetchone()
        if row is None:
            raise KeyError("Model not found")
        return {
            "spec": json.loads(row["spec"]),
            "head": row["head"],
            "original": row["original"],
            "default_input": json.loads(row["default_input"]),
            "head_name": row["head_name"],
            "original_name": row["original_name"],
        }

    def revision(self, revision_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM revisions WHERE id=?", (revision_id,)).fetchone()
        if row is None:
            raise KeyError("Revision not found")
        result = dict(row)
        result["tensors"] = json.loads(result["tensors"])
        result["changes"] = json.loads(result["changes"])
        return result

    def snapshot(self, model_id: str, revision_id: str):
        model, revision = self.model(model_id), self.revision(revision_id)
        if revision["model_id"] != model_id:
            raise ValueError("Revision belongs to another model")
        return ModelSpec.model_validate(model["spec"]), {
            key: self.read_tensor(digest) for key, digest in revision["tensors"].items()
        }

    def edit(self, model_id: str, request: EditRequest) -> dict:
        payload = request.model_dump() if request.name else request.model_dump(exclude={"name"})
        fingerprint = hashlib.sha256(canonical(payload).encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT * FROM requests WHERE model_id=? AND key=?",
                (model_id, request.idempotency_key),
            ).fetchone()
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise ConflictError("Idempotency key was already used for a different edit")
                return self.revision(previous["revision_id"])
            model = self.model(model_id)
            if model["head"] != request.base_revision:
                raise ConflictError("Active revision changed; refresh before applying this edit")
            spec, parameters = self.snapshot(model_id, request.base_revision)
            parameters, changes = apply_changes(spec, parameters, request.changes)
            if not changes:
                raise ValueError("No effective parameter changes to save")
            blobs = self.revision(request.base_revision)["tensors"]
            for key in {change["tensor_id"] for change in changes}:
                blobs[key] = self.write_tensor(parameters[key])
            revision_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO revisions (id, model_id, parent, tensors, changes, created_at, name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    revision_id,
                    model_id,
                    request.base_revision,
                    canonical(blobs),
                    canonical(changes),
                    datetime.now(UTC).isoformat(),
                    request.name or f"Revision {revision_id[:8]}",
                ),
            )
            db.execute("UPDATE models SET head=? WHERE id=?", (revision_id, model_id))
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?)",
                (model_id, request.idempotency_key, fingerprint, revision_id),
            )
        return self.revision(revision_id)

    def checkout(self, model_id: str, expected_head: str, revision_id: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = self.revision(revision_id)
            if revision["model_id"] != model_id:
                raise ValueError("Cannot restore a revision from another model")
            changed = db.execute(
                "UPDATE models SET head=? WHERE id=? AND head=?",
                (revision_id, model_id, expected_head),
            ).rowcount
            if changed != 1:
                raise ConflictError("Active revision changed; refresh before restoring")
        return self.model(model_id)

    def history(self, model_id: str) -> list[dict]:
        self.model(model_id)
        with self.connection() as db:
            ids = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM revisions WHERE model_id=? ORDER BY rowid DESC LIMIT 100",
                    (model_id,),
                )
            ]
        return [self.revision(revision) for revision in ids]

    def record_run(self, result: dict) -> dict:
        run = {**result, "id": uuid.uuid4().hex}
        with self.connection() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?)",
                (run["id"], run["revision_id"], canonical(run), datetime.now(UTC).isoformat()),
            )
        return run
