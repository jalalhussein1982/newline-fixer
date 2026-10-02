# syntax=docker/dockerfile:1
# Builder: dependencies (CPU-only torch, pinned by pyproject on Linux) and the weights by revision.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ARG NF_MODEL_REVISION=6c311e757d17e89c80b7b86908043637a4f56e28
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv UV_PYTHON_DOWNLOADS=never HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra model --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra model
RUN /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('jalalhussein1982/newline-fixer-scratch', revision='${NF_MODEL_REVISION}', local_dir='/app/weights')" \
    && rm -rf /app/weights/.cache

# Runtime: the virtual environment, the sources it points at, the weights; non-root.
FROM python:3.12-slim-bookworm
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 NF_MODEL=rules NF_WEIGHTS=/app/weights
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/weights /app/weights
RUN useradd --system --uid 10001 --no-create-home app && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "newline_fixer.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
