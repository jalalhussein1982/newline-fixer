"""The FastAPI application (design 6.1). `create_app` wraps one Fixer chosen by Settings."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint
from starlette.routing import Match

from ..models.base import Fixer
from ..windows import fix
from .config import Settings, load_fixer
from .metrics import CONTENT_TYPE, Metrics
from .request_log import configure_logging, log_request
from .schemas import FixRequest, FixResponse, FixStats, Health


class ModelState:
    """The served fixer once loaded, or the reason it is not."""

    def __init__(self) -> None:
        self.fixer: Fixer | None = None
        self.error: str | None = None


def route_template(request: Request) -> str:
    """The matched route's path template, or 'unmatched'; never the raw path (label cardinality)."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return str(getattr(route, "path", "unmatched"))
    return "unmatched"


def create_app(
    settings: Settings | None = None,
    loader: Callable[[Settings], Fixer] | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    load = loader or load_fixer
    state = ModelState()
    configure_logging(cfg.log_level)
    metrics = Metrics()
    metrics.set_model(cfg.model, cfg.weights_source() or "")

    def _load() -> None:
        try:
            state.fixer = load(cfg)
            metrics.set_model(state.fixer.name, cfg.weights_source() or "")
        except Exception as e:  # surfaced on /healthz and /v1/fix, never swallowed
            state.error = f"{type(e).__name__}: {e}"

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        loading = asyncio.get_running_loop().run_in_executor(None, _load)
        yield
        await loading

    app = FastAPI(title="newline-fixer", version="0.1.0", lifespan=lifespan)
    app.state.settings = cfg
    app.state.model = state
    app.state.metrics = metrics

    @app.exception_handler(RequestValidationError)
    async def invalid_body(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the offending input, which cannot be encoded when it is
        # a lone surrogate; report where and why, not the value.
        detail = [{k: v for k, v in e.items() if k in ("type", "loc", "msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        if state.fixer is None:
            body: dict[str, object] = Health(
                status="error" if state.error else "loading", model=cfg.model, ready=False
            ).model_dump()
            if state.error:
                body["error"] = state.error
            return JSONResponse(status_code=503, content=body)
        return JSONResponse(Health(status="ok", model=state.fixer.name, ready=True).model_dump())

    @app.post("/v1/fix", response_model=FixResponse)
    async def fix_text(body: FixRequest, request: Request) -> FixResponse:
        if len(body.text) > cfg.max_chars:
            raise HTTPException(
                413, f"text has {len(body.text)} characters; the limit is {cfg.max_chars}"
            )
        fixer = state.fixer
        if fixer is None:
            reason = f": {state.error}" if state.error else ""
            raise HTTPException(503, f"model not loaded{reason}")
        t0 = time.perf_counter()
        result = await run_in_threadpool(fix, body.text, fixer)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        request.state.input_chars = len(body.text)
        request.state.changed = result.changed
        return FixResponse(
            text=result.text,
            stats=FixStats(
                tokens=result.tokens,
                gaps=result.gaps,
                changed=result.changed,
                model=fixer.name,
                latency_ms=latency_ms,
            ),
        )

    @app.middleware("http")
    async def observe(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            seconds = time.perf_counter() - t0
            endpoint = route_template(request)
            input_chars = getattr(request.state, "input_chars", None)
            changed = getattr(request.state, "changed", None)
            metrics.observe(endpoint, status, seconds, input_chars, changed)
            log_request(
                request_id=request_id,
                endpoint=endpoint,
                status=status,
                input_chars=input_chars,
                changed=changed,
                latency_ms=seconds * 1000,
                model=state.fixer.name if state.fixer else cfg.model,
            )

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(metrics.render(), media_type=CONTENT_TYPE)

    return app
