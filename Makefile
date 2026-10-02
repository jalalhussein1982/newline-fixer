.PHONY: check lint type test fmt serve sync image container-check space-check bundle bundle-check

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

serve:
	uv run uvicorn newline_fixer.service.app:create_app --factory --host 0.0.0.0 --port 8000

image:
	docker build -t newline-fixer:local .

container-check:
	uv run python scripts/container_check.py

SPACE_URL ?= https://jalalhussein1982-newline-fixer.hf.space
space-check:
	uv run python scripts/container_check.py --url $(SPACE_URL) --expect-mismatch

bundle:
	for b in $$(git branch -r | grep -v -- '->' | sed 's|^ *origin/||'); do \
	  git show-ref --verify --quiet "refs/heads/$$b" || git branch --track "$$b" "origin/$$b"; \
	done
	git bundle create jalal-hussein.bundle --all

bundle-check:
	bash scripts/bundle_check.sh
