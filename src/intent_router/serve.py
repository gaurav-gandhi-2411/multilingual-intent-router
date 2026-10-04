"""FastAPI service for the intent router.

    ROUTER_MODEL_DIR=serve_model ROUTER_BACKEND=torch ROUTER_THREADS=4 \
        uvicorn intent_router.serve:app --host 0.0.0.0 --port 8000

(ROUTER_BACKEND defaults to onnx_fp32; torch / onnx_int8 / onnx_int8_encoder are selectable.)
POST /predict  {"text": str (1..2000 chars, not blank)} -> predict() dict + model_version /
               model_fingerprint / backend. GET /health -> 200 only once the model is loaded.
Every error body is {"code": str, "message": str}; request text is never echoed back or logged.
The model is loaded once in the lifespan hook (a failed load aborts startup: fail fast).
Endpoints are plain `def`, so Starlette runs them in its threadpool; the Router serialises the
forward pass with a lock (parallel forwards would only oversubscribe the same CPU threads).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictStr, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from intent_router.predict import Router, env_settings

MAX_TEXT_CHARS = 2000
log = logging.getLogger("intent_router.serve")


class PredictRequest(BaseModel):
    """Body of POST /predict."""

    text: StrictStr = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class TopItem(BaseModel):
    """One of the top-3 classes."""

    label: str
    prob: float


class PredictResponse(BaseModel):
    """Prediction (see intent_router.predict for the exact semantics) plus model identity."""

    label: str
    confidence: float
    ood_score: float
    abstained: bool
    top3: list[TopItem]
    model_version: str
    model_fingerprint: str
    backend: str


class ErrorBody(BaseModel):
    """Uniform error shape."""

    code: str
    message: str


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message})


def create_app() -> FastAPI:
    """Build the app; the router loads from ROUTER_MODEL_DIR/_BACKEND/_THREADS at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model_dir, backend, threads = env_settings()
        t0 = time.perf_counter()
        app.state.router = Router.load(model_dir, backend, threads)
        log.info(
            "model loaded",
            extra={
                "backend": backend,
                "version": app.state.router.cfg.version,
                "load_s": round(time.perf_counter() - t0, 2),
            },
        )
        yield
        app.state.router = None

    app = FastAPI(title="intent-router", lifespan=lifespan)
    app.state.router = None

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Only location + rule text: pydantic's "input" field would echo the request text.
        parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return _error(422, "validation_error", "; ".join(parts))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error", extra={"error_type": type(exc).__name__})
        return _error(500, "internal_error", "internal error; see server logs")

    def _router(request: Request) -> Router | None:
        r: Router | None = request.app.state.router
        return r

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        r = _router(request)
        if r is None:
            return _error(503, "not_ready", "model not loaded")
        return JSONResponse(
            {
                "status": "ok",
                "ready": True,
                "model_version": r.cfg.version,
                "model_fingerprint": r.cfg.model_fingerprint,
                "backend": r.backend_name,
            }
        )

    @app.post("/predict", response_model=PredictResponse, responses={422: {"model": ErrorBody}})
    def predict(body: PredictRequest, request: Request) -> Any:
        r = _router(request)
        if r is None:
            return _error(503, "not_ready", "model not loaded")
        t0 = time.perf_counter()
        pred = r.predict(body.text)
        log.info(
            "predict",
            extra={
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "backend": r.backend_name,
                "abstained": pred["abstained"],
                "n_chars": len(body.text),
            },
        )
        return {
            **pred,
            "model_version": r.cfg.version,
            "model_fingerprint": r.cfg.model_fingerprint,
            "backend": r.backend_name,
        }

    return app


app = create_app()
