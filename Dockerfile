# syntax=docker/dockerfile:1
# Builder: dependencies (CPU-only torch, pinned by pyproject on Linux) and both models' weights by revision.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ARG NF_MODEL_REVISION=6c311e757d17e89c80b7b86908043637a4f56e28
ARG NF_FINETUNED_REVISION=11d6b26e80dfa2c9606702cd2755a63c9dce99ed
ARG WITH_WEIGHTS=1
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv UV_PYTHON_DOWNLOADS=never HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra model --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra model
RUN if [ "$WITH_WEIGHTS" = "1" ]; then \
      /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
      snapshot_download('jalalhussein1982/newline-fixer-scratch', revision='${NF_MODEL_REVISION}', local_dir='/app/weights')" \
      && /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
      snapshot_download('jalalhussein1982/newline-fixer-finetuned', revision='${NF_FINETUNED_REVISION}', local_dir='/app/weights-finetuned')" \
      && rm -rf /app/weights/.cache /app/weights-finetuned/.cache; \
    else mkdir -p /app/weights /app/weights-finetuned; fi

# Runtime: the virtual environment, the sources it points at, both weights directories; non-root.
FROM python:3.12-slim-bookworm
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 NF_MODEL=finetuned NF_WEIGHTS_SCRATCH=/app/weights NF_WEIGHTS_FINETUNED=/app/weights-finetuned
WORKDIR /app
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/weights /app/weights
COPY --from=builder --chown=app:app /app/weights-finetuned /app/weights-finetuned
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["uvicorn", "newline_fixer.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
