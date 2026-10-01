.PHONY: check lint type test fmt sync

# chflags: some macOS setups mark .venv hidden, which makes Python ignore its .pth files
sync:
	uv sync --all-extras
	chflags -R nohidden .venv 2>/dev/null || true

check: lint type test

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

type:
	uv run mypy

test:
	uv run pytest

fmt:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts
