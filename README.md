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
| `src/newline_fixer/service/` | FastAPI app: `POST /v1/fix`, `/healthz`, `/metrics`, demo page at `/` |
| `scripts/` | data building and evaluation entry points |
| `tests/` | pytest suite |
| `data/` | see `data/README.md` |
| `report.md` | Final report: how to run, approach, decisions, results (written last) |

The report arrives with M4.

## Run the service

```bash
uv sync --all-extras
make serve                       # http://localhost:8000, demo page at /
curl -s localhost:8000/healthz
curl -s localhost:8000/v1/fix -H 'content-type: application/json' \
  -d '{"text": "3.2.3 Applications of Attention\n in our Model The Transformer uses multi-head attention in three different ways: • In \"encoder-decoder attention\" layers,\n the que\nries come from the previous decoder layer."}'
```

Configuration, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `NF_MODEL` | `rules` | `identity`, `rules` or `scratch` (decision 0008 sets the default) |
| `NF_MODEL_REVISION` | the published scratch-v1 revision | Hub revision of the scratch weights |
| `NF_WEIGHTS` | unset | a local run directory or `hf:repo@revision`; overrides `NF_MODEL_REVISION` |
| `NF_MAX_CHARS` | `100000` | inputs longer than this get 413 |
| `NF_LOG_LEVEL` | `INFO` | level of the JSON request log on stdout |
| `NF_DEVICE` | `cpu` | torch device for the scratch model |

`GET /healthz` answers 503 until the model is loaded; `GET /metrics` is Prometheus text. One JSON line per request goes to stdout; request text is never logged.

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

Weights live on the Hugging Face Hub, not in git (decision 0003). Published: `scratch-v1` at https://huggingface.co/jalalhussein1982/newline-fixer-scratch, revision `6c311e757d17e89c80b7b86908043637a4f56e28`. `NF_WEIGHTS=hf:jalalhussein1982/newline-fixer-scratch@6c311e757d17e89c80b7b86908043637a4f56e28` makes the service download and use that exact revision; the repo id and revision are recorded under the `hub` key of `experiments/training/scratch-v1.json`. To publish (after `hf auth login`): `uv run python scripts/publish_weights.py --run-id scratch-v1 --repo <user>/newline-fixer-scratch`.

Results tables live in `experiments/README.md`.
