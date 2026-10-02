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

The from-scratch model needs the `model` extra (PyTorch); `uv sync --all-extras` installs it. Training uses CUDA or Apple MPS when available and falls back to CPU. On the M1 a full eight-epoch run takes about 100 minutes, so the two Task 7 runs are meant for a free Colab GPU: open `notebooks/train_scratch_colab.ipynb` in Colab, which clones this repository, trains both runs and hands back `experiments/runs/` and `experiments/training/` as a zip.

On macOS, if `uv run python -c 'import newline_fixer'` fails with ModuleNotFoundError, run `make sync`: some setups mark `.venv` hidden and Python 3.12+ then ignores its `.pth` files.

Rebuild data: see the docstring of `scripts/build_data.py`. Evaluate:

```bash
uv run python scripts/evaluate.py --systems identity,rules --sets V1,V2,V3 --out experiments/results/dev.json
uv run python scripts/results_table.py
```

Train the from-scratch model:

```bash
uv run python scripts/train_scratch.py --run-id scratch-v1 --seed 1 --class-weights none
uv run python scripts/evaluate.py --systems identity,rules,scratch --sets V1,V2,V3 --out experiments/results/m2-scratch.json
```

Weights are written to `experiments/runs/<run-id>/` (git-ignored); set `NF_WEIGHTS` to a run directory to evaluate or serve it. The two runs behind decision 0007 (`scratch-v1`, and `scratch-v1-inverse` with `--class-weights inverse`) were produced with the Colab notebook above, and `experiments/runs/current` is a copy of `scratch-v1`.

Results tables live in `experiments/README.md`.
