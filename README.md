# newline-fixer

A machine learning service that fixes newline placement in English text.
Built for the BottleCapAI Applied ML Engineer challenge.

This repository tracks the whole project from the first written requirement to the
final service. The history is intentional: the planning documents under `docs/`
were committed before any code, and every later change to them is a commit.

## Layout

| Path | Purpose |
|---|---|
| `docs/01-requirements.md` | What the service must do, what "fixed newlines" means, acceptance criteria |
| `docs/02-design.md` | How it is built: data, model, baselines, evaluation, service (written after the requirements are agreed) |
| `docs/03-implementation-plan.md` | Ordered, testable tasks derived from the design |
| `docs/decisions/` | Architecture decision records, one file per decision, never rewritten |
| `src/newline_fixer/` | library |
| `scripts/` | data building and evaluation entry points |
| `tests/` | pytest suite |
| `data/` | see `data/README.md` |
| `report.md` | Final report: how to run, approach, decisions, results (written last) |

Code, tests, the Dockerfile and the report arrive in later commits.

## Development

```bash
uv sync --all-extras   # creates .venv with all dependencies
make check             # lint, type check, tests
```

The from-scratch model needs the `model` extra (PyTorch); `uv sync --all-extras` installs it. Training uses Apple MPS when available and falls back to CPU.

On macOS, if `uv run python -c 'import newline_fixer'` fails with ModuleNotFoundError, run `make sync`: some setups mark `.venv` hidden and Python 3.12+ then ignores its `.pth` files.

Rebuild data: see the docstring of `scripts/build_data.py`. Evaluate:

```bash
uv run python scripts/evaluate.py --systems identity,rules --sets V1,V2,V3 --out experiments/results/dev.json
uv run python scripts/results_table.py
```

Results tables live in `experiments/README.md`.
