"""Local API for supported dense models and native data-only packages."""

import hashlib
import os
import secrets
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from neurolabs.diagnostics import analyze
from neurolabs.editing import apply_changes, draft_id
from neurolabs.imports import ImportBusyError, ImportPreviews
from neurolabs.packages import MAX_PACKAGE_BYTES, Examples, Manifest, make_package, read_package
from neurolabs.runtime import execute
from neurolabs.samples import samples
from neurolabs.schema import (
    CheckoutRequest,
    DiagnosticRequest,
    DraftPreviewRequest,
    EditRequest,
    ImportCommitRequest,
    ModelSpec,
    RunRequest,
)
from neurolabs.store import ConflictError, Store


def data_directory() -> Path:
    if value := os.environ.get("NEUROLABS_DATA_DIR") or os.environ.get("NEUROLENS_DATA_DIR"):
        return Path(value)
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    current = base / "neuroLabs"
    legacy = base / "NeuroLens"
    if not (current / "neurolabs.sqlite3").exists() and (legacy / "neurolens.sqlite3").is_file():
        return legacy
    return current


def create_app(root: Path | None = None) -> FastAPI:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    store = Store(root or data_directory())
    previews = ImportPreviews()
    for spec, parameters, default_input in samples():
        store.register(spec, parameters, default_input)
    app = FastAPI(title="neuroLabs", version="0.1.0")
    session_token = secrets.token_urlsafe(32)
    app.state.store = store
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        # No CORS wildcard: reject cross-origin requests, including localhost from other ports.
        origin = request.headers.get("origin")
        permitted = {
            f"http://{request.headers.get('host')}",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        }
        if origin is not None and origin not in permitted:
            return JSONResponse(
                {
                    "error": {
                        "code": "origin_rejected",
                        "message": "Use the local neuroLabs application",
                    }
                },
                403,
            )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse(
                {"error": {"code": "origin_rejected", "message": "Cross-site access is disabled"}},
                403,
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not secrets.compare_digest(
            request.headers.get("x-neurolabs-session")
            or request.headers.get("x-neurolens-session", ""),
            session_token,
        ):
            return JSONResponse(
                {
                    "error": {
                        "code": "session_required",
                        "message": "Refresh the local session and retry",
                    }
                },
                403,
            )
        return await call_next(request)

    @app.exception_handler(ConflictError)
    async def conflict_handler(request, exc):
        return JSONResponse({"error": {"code": "revision_conflict", "message": str(exc)}}, 409)

    @app.exception_handler(ImportBusyError)
    async def import_busy_handler(request, exc):
        return JSONResponse({"error": {"code": "import_busy", "message": str(exc)}}, 503)

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        return JSONResponse({"error": {"code": "invalid_operation", "message": str(exc)}}, 422)

    @app.exception_handler(KeyError)
    async def missing_handler(request, exc):
        return JSONResponse({"error": {"code": "not_found", "message": str(exc)}}, 404)

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "runtime": "pytorch-cpu",
            "capabilities": [
                "samples",
                "execute",
                "capture",
                "edit",
                "restore",
                "native-import",
                "native-export",
                "onnx-import",
            ],
        }

    @app.get("/api/v1/session")
    def session():
        return JSONResponse({"token": session_token}, headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/imports/preview")
    async def import_preview(
        request: Request,
        source_format: Literal["zip", "onnx"] = Query("zip", alias="format"),
        filename: str = Query("model.onnx", max_length=255),
    ):
        # Stream with a hard cap, including clients without Content-Length. No multipart parser.
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_PACKAGE_BYTES:
            return JSONResponse({"error": {"message": "Model packages must be at most 6 MiB"}}, 413)
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_PACKAGE_BYTES:
                return JSONResponse(
                    {"error": {"message": "Model packages must be at most 6 MiB"}}, 413
                )
            data.extend(chunk)
        return await run_in_threadpool(previews.inspect, bytes(data), source_format, filename)

    @app.post("/api/v1/imports/{token}", status_code=201)
    def import_model(token: str, options: ImportCommitRequest | None = None):
        options = options or ImportCommitRequest()
        data, result = previews.get(token)
        manifest, values, _ = read_package(data)
        # Server-assigned identity avoids overwriting samples or another uploaded model.
        spec = ModelSpec.model_validate(
            {
                **manifest.spec.model_dump(),
                "id": f"import-{token}",
                "name": options.name or manifest.spec.name,
                "task": options.task or manifest.spec.task,
            }
        )
        store.register(
            spec,
            values,
            list(manifest.default_input),
            import_info={
                "package_sha256": result["sha256"],
                "metadata": manifest.metadata.model_dump(),
                "verification": result["verification"],
                "source_format": result.get("source_format", "zip"),
                "source_sha256": result.get("source_sha256", result["sha256"]),
                "source_filename": result.get("source_filename"),
            },
        )
        return store.model(spec.id)

    @app.get("/api/v1/import-example")
    def import_example():
        spec, values, inputs = next(item for item in samples() if item[0].id == "curve")
        spec = spec.model_copy(update={"name": "Imported rectified curve"})
        rows = [[-1.0], [0.5], [1.0]]
        examples = Examples(inputs=rows, outputs=execute(spec, values, rows, "example")["outputs"])
        manifest = Manifest(spec=spec, default_input=inputs)
        return Response(
            make_package(manifest, values, examples),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="neurolabs-example.zip"'},
        )

    @app.get("/api/v1/import-example.onnx")
    def import_onnx_example():
        from neurolabs.onnx_import import make_onnx_example

        spec, values, _ = next(item for item in samples() if item[0].id == "curve")
        return Response(
            make_onnx_example(spec, values),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="neurolabs-example.onnx"'},
        )

    @app.get("/api/v1/models")
    def list_models():
        return store.models()

    @app.get("/api/v1/models/{model_id}")
    def get_model(model_id: str):
        return store.model(model_id)

    @app.get("/api/v1/models/{model_id}/revisions")
    def revisions(model_id: str):
        return store.history(model_id)

    @app.get("/api/v1/models/{model_id}/revisions/{revision_id}/parameters")
    def parameters(model_id: str, revision_id: str):
        _, values = store.snapshot(model_id, revision_id)
        if sum(value.numel() for value in values.values()) > 100_000:
            raise ValueError("Use parameter slices for models larger than 100,000 parameters")
        return {
            "revision_id": revision_id,
            "tensors": {
                key: {"shape": list(value.shape), "values": value.tolist()}
                for key, value in values.items()
            },
        }

    @app.get("/api/v1/models/{model_id}/revisions/{revision_id}/tensors/{tensor_id}")
    def tensor_slice(
        model_id: str,
        revision_id: str,
        tensor_id: str,
        row: int = Query(0, ge=0),
        rows: int = Query(64, ge=1, le=64),
        column: int = Query(0, ge=0),
        columns: int = Query(64, ge=1, le=64),
    ):
        _, values = store.snapshot(model_id, revision_id)
        if tensor_id not in values:
            raise KeyError("Tensor not found")
        tensor = values[tensor_id]
        if row >= tensor.shape[0] or column >= (tensor.shape[1] if tensor.ndim == 2 else 1):
            raise ValueError("Slice starts outside the tensor")
        sliced = tensor[row : row + rows]
        if tensor.ndim == 2:
            sliced = sliced[:, column : column + columns]
        return {
            "revision_id": revision_id,
            "tensor_id": tensor_id,
            "shape": list(tensor.shape),
            "row": row,
            "column": column,
            "values": sliced.tolist(),
        }

    @app.get("/api/v1/models/{model_id}/revisions/{revision_id}/parameter-data")
    def parameter_data(model_id: str, revision_id: str):
        spec, values = store.snapshot(model_id, revision_id)
        # At most 4 MB at the enforced 1M float32 limit; order comes from immutable spec.
        data = b"".join(
            values[tensor.id].numpy().astype("<f4").tobytes() for tensor in spec.tensors
        )
        return Response(
            data,
            media_type="application/octet-stream",
            headers={
                "ETag": hashlib.sha256(data).hexdigest(),
                "X-Revision-ID": revision_id,
                "Cache-Control": "private, max-age=31536000, immutable",
            },
        )

    @app.get("/api/v1/models/{model_id}/revisions/{revision_id}/export")
    def export_model(model_id: str, revision_id: str):
        spec, values = store.snapshot(model_id, revision_id)
        model = store.model(model_id)
        info = model["import_info"] or {}
        manifest = Manifest(
            spec=spec, default_input=model["default_input"], metadata=info.get("metadata", {})
        )
        rows = [model["default_input"]]
        examples = Examples(
            inputs=rows, outputs=execute(spec, values, rows, revision_id)["outputs"]
        )
        data = make_package(manifest, values, examples)
        return Response(
            data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{model_id}-{revision_id[:8]}.zip"',
            },
        )

    @app.post("/api/v1/models/{model_id}/revisions", status_code=201)
    def edit(model_id: str, request: EditRequest):
        return store.edit(model_id, request)

    @app.post("/api/v1/models/{model_id}/checkout")
    def checkout(model_id: str, request: CheckoutRequest):
        return store.checkout(model_id, request.expected_head, request.revision_id)

    @app.post("/api/v1/models/{model_id}/runs")
    def run(model_id: str, request: RunRequest):
        spec, values = store.snapshot(model_id, request.revision_id)
        return store.record_run(
            execute(spec, values, request.inputs, request.revision_id, request.captures)
        )

    @app.post("/api/v1/models/{model_id}/diagnostics")
    def diagnostics(model_id: str, request: DiagnosticRequest):
        spec, values = store.snapshot(model_id, request.revision_id)
        _, baseline = store.snapshot(model_id, request.baseline_revision_id)
        values, changes = apply_changes(spec, values, request.changes)
        revision = draft_id(request.revision_id, changes) if changes else request.revision_id
        report = analyze(
            spec,
            values,
            baseline,
            request.inputs,
            revision,
            request.baseline_revision_id,
        )
        if changes:
            report.update(
                draft_base_revision_id=request.revision_id,
                draft_changes=changes,
                draft_name=request.draft_name,
            )
        report["revision_name"] = (
            request.draft_name or "Unsaved draft"
            if changes
            else store.revision(request.revision_id)["name"]
        )
        report["baseline_revision_name"] = store.revision(request.baseline_revision_id)["name"]
        report["base_revision_name"] = store.revision(request.revision_id)["name"]
        return report

    @app.post("/api/v1/models/{model_id}/draft-preview")
    def preview(model_id: str, request: DraftPreviewRequest):
        spec, values = store.snapshot(model_id, request.base_revision)
        values, changes = apply_changes(spec, values, request.changes)
        revision = draft_id(request.base_revision, changes)
        captures = (spec.input_name, *(op.output for op in spec.operations))
        rows = [[-2 + index / 16, *request.inputs[1:]] for index in range(65)]
        run = execute(spec, values, [request.inputs], revision, captures)
        run.update(id=revision, is_draft=True, base_revision=request.base_revision)
        try:
            sweep = execute(spec, values, rows, revision)
            sweep.update(id=revision, is_draft=True, base_revision=request.base_revision)
        except ValueError:
            # A fixed visualization sweep can overflow while the selected input is valid.
            sweep = None
        return {"run": run, "sweep": sweep}

    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
