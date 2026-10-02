"""The FastAPI application (design 6.1). `create_app` wraps one Fixer chosen by Settings."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..models.base import Fixer
from ..windows import fix
from .config import Settings, load_fixer
from .schemas import FixRequest, FixResponse, FixStats, Health


class ModelState:
    """The served fixer once loaded, or the reason it is not."""

    def __init__(self) -> None:
        self.fixer: Fixer | None = None
        self.error: str | None = None


def create_app(
    settings: Settings | None = None,
    loader: Callable[[Settings], Fixer] | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    load = loader or load_fixer
    state = ModelState()

    def _load() -> None:
        try:
            state.fixer = load(cfg)
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

    return app
