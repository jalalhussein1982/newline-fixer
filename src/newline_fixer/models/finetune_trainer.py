"""Fine-tuning loop for the pretrained encoder (design 4.5, 4.7)."""

from __future__ import annotations

import datetime as dt
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..data.records import CleanDoc, EvalItem
from .finetune_encoding import Encoded, collate, encode_window, labels_for
from .finetuned import FinetunedConfig, FinetunedFixer
from .train_data import Example, class_counts, epoch_examples
from .trainer import _git_commit, dev_metrics

Builder = Callable[[FinetunedConfig, torch.device], FinetunedFixer]
Item = tuple[Encoded, list[int]]


@dataclass(frozen=True)
class FinetuneTrainConfig:
    run_id: str
    pretrained: str
    random_init: bool = False
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06
    patience: int = 1
    seed: int = 1
    grad_clip: float = 1.0
    max_docs: int | None = None
    max_examples: int | None = None
    token_budget: int = 192


def _encode(examples: Sequence[Example], fixer: FinetunedFixer) -> tuple[list[Item], int]:
    items: list[Item] = []
    overflow = 0
    for e in examples:
        enc = encode_window(e.tokens, e.current, fixer.tokenizer, fixer.cfg.max_len)
        overflow += int(enc.overflowed)
        items.append((enc, labels_for(enc, e.target)))
    return items, overflow


def train_finetune(
    tcfg: FinetuneTrainConfig,
    train_docs: Sequence[CleanDoc],
    dev: dict[str, list[EvalItem]],
    run_dir: Path,
    device: torch.device,
    build: Builder = FinetunedFixer.from_pretrained_name,
) -> dict[str, object]:
    if "V1" not in dev:
        raise ValueError("dev must contain 'V1': the model is selected by V1 macro-F1")
    docs = list(train_docs)[: tcfg.max_docs] if tcfg.max_docs else list(train_docs)
    torch.manual_seed(tcfg.seed)
    torch.cuda.manual_seed_all(tcfg.seed)
    cfg = FinetunedConfig(pretrained=tcfg.pretrained, random_init=tcfg.random_init)
    fixer = build(cfg, device)
    model = fixer.model
    pad_id = int(fixer.tokenizer.pad_token_id)

    def examples_for(epoch: int) -> list[Example]:
        out = epoch_examples(docs, tcfg.seed, epoch, tcfg.token_budget)
        return out[: tcfg.max_examples] if tcfg.max_examples else out

    first_epoch = examples_for(0)
    counts = class_counts(first_epoch)
    first_items, n_overflow = _encode(first_epoch, fixer)

    # Later epochs re-corrupt the same documents, so their example count is close to
    # epoch 0's; the schedule is sized from epoch 0 and the last steps clamp at zero.
    total_steps = max(1, tcfg.epochs * math.ceil(len(first_items) / tcfg.batch_size))
    warmup = max(1, round(tcfg.warmup_fraction * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    record: dict[str, object] = {
        "run_id": tcfg.run_id,
        "git_commit": _git_commit(),
        "started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": tcfg.seed,
        "pretrained": tcfg.pretrained,
        "random_init": tcfg.random_init,
        "config": asdict(cfg),
        "train_config": asdict(tcfg),
        "n_train_docs": len(docs),
        "n_examples_epoch0": len(first_items),
        "class_counts_epoch0": counts,
        "n_overflow_examples": n_overflow,
        "n_params": sum(p.numel() for p in model.parameters()),
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
        if epoch == 0:
            items, overflow = first_items, n_overflow
        else:
            items, overflow = _encode(examples_for(epoch), fixer)
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for i in range(0, len(items), tcfg.batch_size):
            batch = collate(items[i : i + tcfg.batch_size], pad_id, device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.item())
            steps += 1
        model.eval()
        metrics = dev_metrics(fixer, dev)
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total / max(steps, 1),
            "overflow": overflow,
            "seconds": time.time() - t0,
            **metrics,
        }
        epochs.append(row)
        print(json.dumps(row))
        f1 = metrics["V1_macro_f1"]
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
