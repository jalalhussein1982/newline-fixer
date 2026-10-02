"""Training loop for the from-scratch model (design 4.4, 4.7)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ..data.records import CleanDoc, EvalItem
from ..eval.runner import evaluate_set
from ..text import split
from .scratch import ScratchFixer
from .scratch_config import ScratchConfig
from .scratch_net import GapTagger, count_parameters
from .train_data import IGNORE, batches, class_counts, epoch_examples
from .vocab import CharVocab, WordVocab


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    epochs: int = 8
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.01
    patience: int = 2
    seed: int = 1
    class_weights: str = "none"
    grad_clip: float = 1.0
    max_docs: int | None = None
    char_min_count: int = 5


def class_weight_tensor(counts: Sequence[int], mode: str, device: torch.device) -> Tensor | None:
    if mode == "none":
        return None
    if mode != "inverse":
        raise ValueError(f"unknown class weighting {mode!r}; use 'none' or 'inverse'")
    total = sum(counts)
    raw = [total / (len(counts) * c) if c else 1.0 for c in counts]
    mean = sum(raw) / len(raw)
    return torch.tensor([r / mean for r in raw], dtype=torch.float32, device=device)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _dev_metrics(fixer: ScratchFixer, dev: dict[str, list[EvalItem]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, items in dev.items():
        r = evaluate_set(fixer, items)
        gap = r["gap"]
        assert isinstance(gap, dict)
        out[f"{name}_macro_f1"] = float(gap["macro_f1"])
        out[f"{name}_damage"] = float(gap["damage_rate"])
        out[f"{name}_wrong_join_per_1000"] = float(gap["wrong_join_per_1000"])
    return out


def train(
    cfg: ScratchConfig,
    tcfg: TrainConfig,
    train_docs: Sequence[CleanDoc],
    dev: dict[str, list[EvalItem]],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    docs = list(train_docs)[: tcfg.max_docs] if tcfg.max_docs else list(train_docs)
    tokens = [t for d in docs for t in split(d.clean)[0]]
    words = WordVocab.build(tokens, cfg.vocab_size)
    chars = CharVocab.build(tokens, tcfg.char_min_count)
    torch.manual_seed(tcfg.seed)
    net = GapTagger(cfg, len(words), len(chars)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    first_epoch = epoch_examples(docs, tcfg.seed, 0, cfg.budget)
    weights = class_weight_tensor(class_counts(first_epoch), tcfg.class_weights, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE)

    record: dict[str, object] = {
        "run_id": tcfg.run_id,
        "git_commit": _git_commit(),
        "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": tcfg.seed,
        "config": asdict(cfg),
        "train_config": asdict(tcfg),
        "n_train_docs": len(docs),
        "n_examples_epoch0": len(first_epoch),
        "class_counts_epoch0": class_counts(first_epoch),
        "n_words": len(words),
        "n_chars": len(chars),
        "n_params": count_parameters(net),
        "device": device.type,
        "epochs": [],
    }
    epochs: list[dict[str, object]] = []
    best_f1, best_epoch, bad, start_all = -1.0, 0, 0, time.time()

    def write_record() -> None:
        record["epochs"] = epochs
        record["best_epoch"] = best_epoch
        record["seconds"] = time.time() - start_all
        record["weights_dir"] = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for epoch in range(tcfg.epochs):
        examples = first_epoch if epoch == 0 else epoch_examples(docs, tcfg.seed, epoch, cfg.budget)
        net.train()
        t0, total, steps = time.time(), 0.0, 0
        for batch, target in batches(
            examples, words, chars, cfg.max_chars, tcfg.batch_size, device
        ):
            optimizer.zero_grad()
            logits = net(batch)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), tcfg.grad_clip)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        net.eval()
        fixer = ScratchFixer(cfg, words, chars, net, device)
        metrics = _dev_metrics(fixer, dev)
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total / max(steps, 1),
            "seconds": time.time() - t0,
            **metrics,
        }
        epochs.append(row)
        print(json.dumps(row))
        f1 = metrics.get("V1_macro_f1", -1.0)
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch + 1, 0
            fixer.save(run_dir)
            record["best"] = dict(metrics)
        else:
            bad += 1
        write_record()
        if bad >= tcfg.patience:
            break
    write_record()
    return record
