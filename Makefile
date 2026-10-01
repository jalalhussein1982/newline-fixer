.PHONY: check lint type test fmt sync

sync:
	uv sync --all-extras

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
