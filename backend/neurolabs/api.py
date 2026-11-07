"""Local API for the sample-model vertical slice. Import is deliberately not exposed yet."""

import os
import secrets
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from neurolabs.diagnostics import analyze
from neurolabs.runtime import execute
from neurolabs.samples import samples
from neurolabs.schema import CheckoutRequest, DiagnosticRequest, EditRequest, RunRequest
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
            "capabilities": ["samples", "execute", "capture", "edit", "restore"],
        }

    @app.get("/api/v1/session")
    def session():
        return JSONResponse({"token": session_token}, headers={"Cache-Control": "no-store"})

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
        # Bundled models only. Bounded slices replace this endpoint before browser model import.
        return {
            "revision_id": revision_id,
            "tensors": {
                key: {"shape": list(value.shape), "values": value.tolist()}
                for key, value in values.items()
            },
        }

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
        return analyze(
            spec,
            values,
            baseline,
            request.inputs,
            request.revision_id,
            request.baseline_revision_id,
        )

    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
